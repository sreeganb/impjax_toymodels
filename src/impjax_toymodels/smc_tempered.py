"""Tempered SMC over a pytree reduced state, driven by `blackjax.smc.tempered`.

The second of BlackJAX's two SMC entry points (the first,
`blackjax.smc.base`, is wrapped by smc_fixed_schedule.py). Both walk the same
tempering path pi_lambda = p0 * L^lambda, but they divide the labour
differently, and the difference is what makes this one worth having:

  * `blackjax.smc.base.step` is the raw Feynman-Kac step. The caller supplies
    a hand-built `update_fn` and `weight_fn` -- which is why
    smc_fixed_schedule.py has to construct a fresh JIT-compiled mutation
    closure at every temperature, a real retracing cost.
  * `blackjax.smc.tempered.build_kernel` takes the *pieces* instead
    (logprior, loglikelihood, an MCMC step/init pair, a resampler) and builds
    the reweighting and mutation itself, with `tempering_param` passed in as
    a traced argument. The kernel is therefore built once and JIT-compiled
    once, and the temperature is data rather than a compile-time constant.

That single change is the fix for the retracing overhead flagged against
smc_fixed_schedule.py, and it is also the layer `blackjax.smc.adaptive_
tempered` is built on -- so smc_adaptive_tempered.py reuses this file's
`build_rmh_mcmc_pair`, mirroring BlackJAX's own module dependency.

This file still owns only the *fixed* ladder: the caller says which lambdas
to visit. Letting the sampler choose them lives next door in
smc_adaptive_tempered.py, so either can be deleted without touching the
other (planning.md's modularity rule). No SMC algorithm is implemented here.
"""

import logging
from typing import Callable, List, Optional, Tuple

import blackjax
import blackjax.smc.resampling as resampling
import blackjax.smc.tempered as tempered
import jax
import jax.numpy as jnp
import numpy as np

from .smc_base_sampler import SCHEDULE_REGISTRY
from .smc_particles import DEFAULT_SCORE_BATCH_SIZE, best_particle_selector, particle_count
from .timing import elapsed_timing, start_timing

logger = logging.getLogger(__name__)

DEFAULT_N_PARTICLES = 100
DEFAULT_N_TEMPERATURE_STEPS = 20
DEFAULT_N_MCMC_STEPS = 10
DEFAULT_SCHEDULE = "linear"


def build_rmh_mcmc_pair(
    proposal_fn: Callable[[jax.Array, dict], dict],
) -> Tuple[Callable, Callable]:
    """Adapt our SO(3)-aware proposal to BlackJAX's tempered-SMC kernel contract.

    `blackjax.smc.tempered` mutates particles through a `(mcmc_step_fn,
    mcmc_init_fn)` pair rather than through a whole update function. Inside
    `blackjax.smc.base.update_and_take_last` the step function is invoked as

        mcmc_step_fn(rng_key, state, tempered_logposterior_fn, **params)

    with `params` coming from the `mcmc_parameters` dict (empty here -- our
    proposal's scales are baked into `proposal_fn` by proposals.build_
    composite, not tuned per particle). So the adapter is just a signature
    shim around `blackjax.rmh.build_kernel()`: no Metropolis logic is written
    here, only the argument threading BlackJAX expects.

    Because every kernel in proposals.py is symmetric by construction, no
    Hastings correction (`proposal_logdensity_fn`) is passed -- the plain
    Metropolis acceptance BlackJAX applies by default is already correct.

    Returns
    -------
    (mcmc_step_fn, mcmc_init_fn) ready to hand to
    `blackjax.smc.tempered.build_kernel` or `adaptive_tempered.build_kernel`.
    """
    rmh_kernel = blackjax.rmh.build_kernel()

    def mcmc_step_fn(rng_key: jax.Array, state, logdensity_fn: Callable):
        return rmh_kernel(rng_key, state, logdensity_fn, proposal_fn)

    return mcmc_step_fn, blackjax.rmh.init


def mean_acceptance_rate(info) -> float:
    """Average RMH acceptance across particles and mutation sweeps for one step.

    BlackJAX returns the inner-kernel diagnostics as
    `info.update_info.is_accepted`, shaped (n_particles, n_mcmc_steps) --
    booleans, one per sweep per particle.
    """
    return float(jnp.mean(info.update_info.is_accepted))


def run_tempered_smc(
    rng_key: jax.Array,
    log_prior_fn: Callable[[dict], jnp.ndarray],
    log_likelihood_fn: Callable[[dict], jnp.ndarray],
    log_prob_fn: Callable[[dict], jnp.ndarray],
    initial_particles: dict,
    proposal_fn: Callable[[jax.Array, dict], dict],
    n_temperature_steps: int = DEFAULT_N_TEMPERATURE_STEPS,
    schedule: str = DEFAULT_SCHEDULE,
    lambdas: Optional[np.ndarray] = None,
    n_mcmc_steps: int = DEFAULT_N_MCMC_STEPS,
    resampling_fn: Callable = resampling.systematic,
    score_batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
    verbose: bool = True,
) -> Tuple[object, List[dict], List[float], np.ndarray]:
    """Run fixed-ladder tempered SMC over a batched pytree particle population.

    Parameters
    ----------
    log_prior_fn, log_likelihood_fn : single-particle theta -> scalar. The
        target at lambda is log_prior + lambda * log_likelihood, so
        log_likelihood_fn must be the *bare* IMP-derived term (-score) with
        no prior mixed in, and log_prior_fn whatever priors.py resolved.
        Note the prior is applied in full at every lambda including 0 -- that
        is the whole point of choosing an informative one.
    log_prob_fn : the full (lambda = 1) log-posterior, used only for
        best-particle tracking and reporting.
    initial_particles : batched pytree, `n_particles` on the leading axis of
        every leaf (see smc_particles.initialize_particles).
    proposal_fn : (key, single theta) -> new theta, the symmetric SO(3)-aware
        kernel from proposals.py.
    lambdas : an explicit temperature ladder, overriding
        `n_temperature_steps`/`schedule`. Must start at 0 and end at 1.
    n_mcmc_steps : RMH mutation sweeps applied to each particle per
        temperature step.

    Returns
    -------
    final_state : blackjax.smc.tempered.TemperedSMCState
    best_thetas : one theta per temperature step (including the initial one),
        the best-scoring particle at that step -- the "only the best model
        gets written out" RMF3 policy in wrapper_impjax.
    best_scores : matching log_prob_fn values.
    lambdas : the ladder actually walked.
    """
    n_particles = particle_count(initial_particles)

    if lambdas is None:
        schedule_fn = SCHEDULE_REGISTRY.get(schedule)
        if schedule_fn is None:
            raise ValueError(f"Unknown schedule {schedule!r}; choose from {list(SCHEDULE_REGISTRY)}")
        lambdas = schedule_fn(n_temperature_steps)
        schedule_label = schedule
    else:
        schedule_label = "explicit ladder"
    lambdas = np.asarray(lambdas, dtype=np.float64)

    mcmc_step_fn, mcmc_init_fn = build_rmh_mcmc_pair(proposal_fn)
    # Built once, outside the loop: `tempering_param` is a traced argument of
    # the kernel, so the whole anneal compiles a single time.
    kernel = jax.jit(
        tempered.build_kernel(
            log_prior_fn,
            log_likelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            resampling_fn,
        ),
        static_argnums=(2,),  # num_mcmc_steps controls a lax.scan length
    )

    best_particle = best_particle_selector(log_prob_fn, batch_size=score_batch_size)

    state = tempered.init(initial_particles)
    best_thetas: List[dict] = []
    best_scores: List[float] = []
    theta0, score0 = best_particle(state.particles)
    best_thetas.append(jax.tree_util.tree_map(np.array, theta0))
    best_scores.append(score0)

    logger.info(
        "Running tempered SMC (blackjax.smc.tempered): %d particles, %d temperature "
        "steps (%s), %d MCMC sweeps/step",
        n_particles,
        len(lambdas) - 1,
        schedule_label,
        n_mcmc_steps,
    )

    timer = start_timing()
    for step_idx in range(1, len(lambdas)):
        rng_key, step_key = jax.random.split(rng_key)
        state, info = kernel(step_key, state, n_mcmc_steps, float(lambdas[step_idx]), {})

        theta_i, score_i = best_particle(state.particles)
        best_thetas.append(jax.tree_util.tree_map(np.array, theta_i))
        best_scores.append(score_i)

        if verbose:
            logger.info(
                "tempered SMC step %3d/%d | lambda=%.4f | best log-post=%.2f | mean accept=%.1f%%",
                step_idx,
                len(lambdas) - 1,
                float(state.tempering_param),
                score_i,
                100 * mean_acceptance_rate(info),
            )

    elapsed = elapsed_timing(timer)
    logger.info(
        "tempered SMC finished: %d particles x %d temperature steps in %.2fs wall / "
        "%.2fs cpu, best log-post=%.2f",
        n_particles,
        len(lambdas) - 1,
        elapsed.wall_time,
        elapsed.cpu_time,
        best_scores[-1],
    )

    return state, best_thetas, best_scores, lambdas
