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
for our pytree particles; rather than modify that already-over-budget file,
`_pytree_batched_scorer` below is a small, local, pytree-native equivalent
of the same batching idea. Everything else that assumed a flat array is
rewritten here to work over an arbitrary pytree state (our
{"quaternions", "translations", "bead_coords"} theta), so the SO(3)-aware
proposal from proposals.py can be used as the mutation kernel. No SMC
algorithm is reimplemented: every tempering/resampling step below is
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
from .timing import elapsed_timing, start_timing

logger = logging.getLogger(__name__)

# Untuned defaults -- deliberately modest so a comparison run finishes
# quickly; see doc/design.tex's SMC section for the reasoning.
DEFAULT_N_PARTICLES = 100
DEFAULT_N_TEMPERATURE_STEPS = 20
DEFAULT_N_MCMC_STEPS = 10
DEFAULT_SCHEDULE = "linear"


def _particle_count(particles) -> int:
    """Number of particles from the leading dimension of any pytree leaf."""
    return jax.tree_util.tree_leaves(particles)[0].shape[0]


def _select_particle(particles, index: int):
    """Pull out a single (unbatched) particle from a batched pytree."""
    return jax.tree_util.tree_map(lambda leaf: leaf[index], particles)


def _pytree_batched_scorer(fn: Callable, batch_size: int) -> Callable:
    """Pytree-native counterpart to smc_base_sampler.batched_vmap: applies
    `fn` (single particle -> scalar) to a batched pytree of particles in
    chunks of `batch_size`, to bound peak memory the same way."""
    scan_batch = jax.jit(lambda batch: jax.vmap(fn)(batch))

    def scored(particles):
        n = _particle_count(particles)
        if n <= batch_size:
            return scan_batch(particles)
        results = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = jax.tree_util.tree_map(lambda leaf: leaf[start:end], particles)
            chunk_result = scan_batch(chunk)
            jax.block_until_ready(chunk_result)
            results.append(chunk_result)
        return jnp.concatenate(results, axis=0)

    return scored


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
    n_particles = _particle_count(initial_particles)

    schedule_fn = SCHEDULE_REGISTRY.get(schedule)
    if schedule_fn is None:
        raise ValueError(f"Unknown schedule {schedule!r}; choose from {list(SCHEDULE_REGISTRY)}")
    lambdas = schedule_fn(n_temperature_steps)

    rmh_kernel = blackjax.rmh.build_kernel()
    score_batched = _pytree_batched_scorer(log_prob_fn, batch_size=score_batch_size)

    def best_particle(particles):
        scores = score_batched(particles)
        idx = int(jnp.argmax(scores))
        return _select_particle(particles, idx), float(scores[idx])

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

    timer = start_timing()
    for step_idx in range(1, len(lambdas)):
        rng_key, step_key = jax.random.split(rng_key)
        lam_prev, lam_curr = lambdas[step_idx - 1], lambdas[step_idx]
        delta_lam = lam_curr - lam_prev

        def _tempered_logdensity(theta, _lam=lam_curr):
            return log_prior_fn(theta) + _lam * log_likelihood_fn(theta)

        @jax.jit
        def update_fn(keys, particles, update_params, _tempered=_tempered_logdensity):
            def _mutate_one(key, particle):
                st = blackjax.rmh.init(particle, _tempered)

                def _body(carry, _):
                    k, s = carry
                    k, subk = jax.random.split(k)
                    s, info = rmh_kernel(subk, s, _tempered, proposal_fn)
                    return (k, s), info.is_accepted

                (_, final_st), accepted = jax.lax.scan(_body, (key, st), jnp.arange(n_mcmc_steps))
                return final_st.position, jnp.mean(accepted)

            new_particles, accept_rates = jax.vmap(_mutate_one)(keys, particles)
            return new_particles, {"acceptance_rate": accept_rates}

        @jax.jit
        def weight_fn(particles, _delta=delta_lam):
            return _delta * jax.vmap(log_likelihood_fn)(particles)

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
