"""Fixed-schedule (untuned) SMC over a pytree reduced state.

This is the "SMC base sampler without any tuning of the parameters" --
a fixed temperature ladder and fixed-sigma mutation kernel, no adaptive
step-size or schedule selection. A separate, not-yet-built file
(smc_adaptive_tempered.py -- see project memory) is reserved for the
tuned/adaptive-tempered variant; the two must never share a file so one can
be deleted/reworked without touching the other.

Reuses smc_base_sampler.py's schedule functions (linear/geometric/sigmoid)
directly -- they are generic over `n_steps`, nothing flat-vector-specific
about them. Its `batched_vmap` scoring helper, however, assumes a plain
`(n_particles, n_dims)` array (`inputs.shape[0]`) and is NOT reusable as-is
for our pytree particles; smc_particles.py holds the pytree-native
equivalents, shared with the other SMC variants. Everything else that
assumed a flat array is rewritten here to work over an arbitrary pytree
state (our {"quaternions", "translations", "bead_coords"} theta), so the
SO(3)-aware proposal from proposals.py can be used as the mutation kernel.
No SMC algorithm is reimplemented: every tempering/resampling step below is
exactly `blackjax.smc.base.step`, and every mutation sweep is exactly
`blackjax.rmh.build_kernel()`'s kernel function.
"""

import logging
from typing import Callable, List, Tuple

import blackjax
import blackjax.smc.base as smc_base
import blackjax.smc.resampling as resampling
import jax
import jax.numpy as jnp
import numpy as np

from .smc_base_sampler import SCHEDULE_REGISTRY
from .smc_particles import best_particle_selector, particle_count
from .timing import elapsed_timing, start_timing

logger = logging.getLogger(__name__)

# Untuned defaults -- deliberately modest so a comparison run finishes
# quickly; see doc/design.tex's SMC section for the reasoning.
DEFAULT_N_PARTICLES = 100
DEFAULT_N_TEMPERATURE_STEPS = 20
DEFAULT_N_MCMC_STEPS = 10
DEFAULT_SCHEDULE = "linear"


def run_fixed_schedule_smc(
    rng_key: jax.Array,
    log_prior_fn: Callable[[dict], jnp.ndarray],
    log_likelihood_fn: Callable[[dict], jnp.ndarray],
    log_prob_fn: Callable[[dict], jnp.ndarray],
    initial_particles: dict,
    proposal_fn: Callable[[jax.Array, dict], dict],
    n_temperature_steps: int = DEFAULT_N_TEMPERATURE_STEPS,
    schedule: str = DEFAULT_SCHEDULE,
    n_mcmc_steps: int = DEFAULT_N_MCMC_STEPS,
    resampling_fn: Callable = resampling.systematic,
    score_batch_size: int = 16,
    verbose: bool = True,
) -> Tuple[object, List[dict], List[float], np.ndarray]:
    """Run base (fixed-schedule) SMC over a batched pytree particle population.

    Parameters
    ----------
    log_prior_fn, log_likelihood_fn : single-particle theta -> scalar. The
        tempered target at lambda is log_prior + lambda * log_likelihood
        (doc/design.tex's SMC section); log_likelihood_fn should be the
        IMP-derived term (-score), log_prior_fn whatever prior the caller
        supplies (0 if none).
    log_prob_fn : single-particle theta -> scalar, the full (lambda=1)
        log-posterior, used only for best-particle tracking/reporting.
    initial_particles : pytree with an `n_particles` leading dimension on
        every leaf (e.g. {"quaternions": (n_particles, K, 4), ...}).
    proposal_fn : (key, single-particle theta) -> new single-particle theta;
        the SO(3)-aware kernel from proposals.py. Must be symmetric, same
        contract as custom_rmh.run_custom_proposal_rmh's proposal_fn.
    n_temperature_steps, schedule, n_mcmc_steps : fixed-schedule controls;
        see module-level DEFAULT_* for this file's untuned defaults.

    Returns
    -------
    final_state : blackjax.smc.base.SMCState
    best_thetas : list of single-particle theta dicts, one per temperature
        step (including the initial one) -- the best-scoring particle at
        that step, for the "only the best model gets written out" RMF3
        policy in wrapper_impjax.run_smc_sampling.
    best_scores : matching list of log_prob_fn values.
    lambdas : the fixed temperature schedule actually used.
    """
    n_particles = particle_count(initial_particles)

    schedule_fn = SCHEDULE_REGISTRY.get(schedule)
    if schedule_fn is None:
        raise ValueError(f"Unknown schedule {schedule!r}; choose from {list(SCHEDULE_REGISTRY)}")
    lambdas = schedule_fn(n_temperature_steps)

    rmh_kernel = blackjax.rmh.build_kernel()
    best_particle = best_particle_selector(log_prob_fn, batch_size=score_batch_size)

    state = smc_base.init(initial_particles, {})
    best_thetas: List[dict] = []
    best_scores: List[float] = []
    theta0, score0 = best_particle(state.particles)
    best_thetas.append(jax.tree_util.tree_map(np.array, theta0))
    best_scores.append(score0)

    logger.info(
        "Running fixed-schedule SMC: %d particles, %d temperature steps (%s), "
        "%d MCMC sweeps/step",
        n_particles,
        n_temperature_steps,
        schedule,
        n_mcmc_steps,
    )

    # Both compiled functions are built once, here, outside the temperature
    # loop -- and the inverse-temperature they depend on is passed in as a
    # traced argument rather than closed over as a constant.
    #
    # Defining them inside the loop instead (as this did) creates a fresh
    # Python function object on every temperature step, and JAX caches
    # compiled code by function identity: every single step then paid a full
    # recompile of the vmapped RMH scan. The symptom is a flat per-step wall
    # time where a working cache would make step 1 expensive and the rest
    # nearly free. It is also why this sampler was an order of magnitude
    # slower than smc_tempered.py, which has always built its kernel once.

    def _tempered_logdensity(theta, lam):
        """p(theta) * p(D|theta)^lam, in logs -- the tempered target."""
        return log_prior_fn(theta) + lam * log_likelihood_fn(theta)

    @jax.jit
    def mutate_fn(keys, particles, lam):
        """One MCMC sweep block per particle, at inverse temperature `lam`."""
        def _mutate_one(key, particle):
            def logdensity(theta):
                return _tempered_logdensity(theta, lam)

            st = blackjax.rmh.init(particle, logdensity)

            def _body(carry, _):
                k, s = carry
                k, subk = jax.random.split(k)
                s, info = rmh_kernel(subk, s, logdensity, proposal_fn)
                return (k, s), info.is_accepted

            (_, final_st), accepted = jax.lax.scan(_body, (key, st), jnp.arange(n_mcmc_steps))
            return final_st.position, jnp.mean(accepted)

        new_particles, accept_rates = jax.vmap(_mutate_one)(keys, particles)
        return new_particles, {"acceptance_rate": accept_rates}

    @jax.jit
    def reweight_fn(particles, delta_lam):
        """Incremental log weight for a step of size `delta_lam`."""
        return delta_lam * jax.vmap(log_likelihood_fn)(particles)

    timer = start_timing()
    for step_idx in range(1, len(lambdas)):
        rng_key, step_key = jax.random.split(rng_key)
        lam_prev, lam_curr = float(lambdas[step_idx - 1]), float(lambdas[step_idx])
        delta_lam = lam_curr - lam_prev

        # blackjax calls update_fn(keys, particles, update_params) and
        # weight_fn(particles), so this step's lambda has to be bound
        # somewhere -- into these two throwaway adapters, never into the
        # compiled functions above.
        def update_fn(keys, particles, update_params, _lam=lam_curr):
            return mutate_fn(keys, particles, _lam)

        def weight_fn(particles, _delta=delta_lam):
            return reweight_fn(particles, _delta)

        state, info = smc_base.step(step_key, state, update_fn, weight_fn, resampling_fn)

        theta_i, score_i = best_particle(state.particles)
        best_thetas.append(jax.tree_util.tree_map(np.array, theta_i))
        best_scores.append(score_i)

        if verbose:
            mean_accept = float(jnp.mean(info.update_info["acceptance_rate"]))
            logger.info(
                "SMC step %3d/%d | lambda=%.4f | best log-post=%.2f | mean accept=%.1f%%",
                step_idx,
                n_temperature_steps,
                lam_curr,
                score_i,
                100 * mean_accept,
            )

    elapsed = elapsed_timing(timer)
    logger.info(
        "fixed-schedule SMC finished: %d particles x %d temperature steps in "
        "%.2fs wall / %.2fs cpu, best log-post=%.2f",
        n_particles,
        n_temperature_steps,
        elapsed.wall_time,
        elapsed.cpu_time,
        best_scores[-1],
    )

    return state, best_thetas, best_scores, lambdas
