"""Adaptive-tempered SMC: the sampler chooses its own temperature ladder.

The tuned counterpart to smc_fixed_schedule.py and smc_tempered.py, wrapping
`blackjax.smc.adaptive_tempered`. A fixed ladder is the thing that makes
untuned SMC slow: a linear schedule spends most of its temperature steps
where the distribution barely changes and then jumps through the region where
it changes fastest, so it is simultaneously wasteful and inaccurate. Here the
next temperature is instead *solved for* at every step:

    given the current particles, find the largest step delta such that the
    effective sample size after reweighting stays at `target_ess * n_particles`

which BlackJAX does with `blackjax.smc.ess.ess_solver` and a dichotomy root
finder. Steps are large where the tempered target is flat and automatically
shrink where it is sharp, so the anneal takes the temperature steps it needs
and no more -- typically an order of magnitude fewer than a hand-set linear
ladder covering the same path.

Consequences for the caller, both deliberate:

  * the number of steps is **not known in advance**. The loop runs until the
    tempering parameter reaches 1 (or `max_steps` trips, which indicates the
    anneal is stuck, not that it finished). So the returned ladder has
    whatever length the run needed, and the RMF3 trajectory has that many
    frames.
  * `target_ess` replaces `n_temperature_steps` as the knob to turn. Lower it
    for a faster, coarser anneal; raise it (towards 1) for a slower, more
    careful one.

Everything below `run_adaptive_tempered_smc` is BlackJAX's: the ESS solve,
the reweighting, the resampling and the RMH mutation sweeps. This file only
drives the loop and tracks the best particle. It reuses smc_tempered.py's
`build_rmh_mcmc_pair`, mirroring BlackJAX's own layering of
`adaptive_tempered` on top of `tempered`.
"""

import logging
from typing import Callable, List, Tuple

import blackjax.smc.adaptive_tempered as adaptive_tempered
import blackjax.smc.resampling as resampling
import blackjax.smc.tempered as tempered
import jax
import jax.numpy as jnp
import numpy as np

from .smc_particles import DEFAULT_SCORE_BATCH_SIZE, best_particle_selector, particle_count
from .smc_tempered import build_rmh_mcmc_pair, mean_acceptance_rate
from .timing import elapsed_timing, start_timing

logger = logging.getLogger(__name__)

DEFAULT_N_PARTICLES = 100
DEFAULT_N_MCMC_STEPS = 10
# Fraction of the population the ESS is held at when picking the next
# temperature. 0.5 is the standard choice in the SMC literature (Del Moral et
# al. 2012): high enough that reweighting does not collapse the population
# onto a handful of particles, low enough to take useful strides.
DEFAULT_TARGET_ESS = 0.5
# Safety stop. The anneal should reach lambda = 1 in far fewer steps than
# this; hitting it means the ESS solver is being forced into vanishing steps,
# which is a signal about the model (or the proposal scales), not a schedule
# to accept silently.
DEFAULT_MAX_STEPS = 200


def run_adaptive_tempered_smc(
    rng_key: jax.Array,
    log_prior_fn: Callable[[dict], jnp.ndarray],
    log_likelihood_fn: Callable[[dict], jnp.ndarray],
    log_prob_fn: Callable[[dict], jnp.ndarray],
    initial_particles: dict,
    proposal_fn: Callable[[jax.Array, dict], dict],
    target_ess: float = DEFAULT_TARGET_ESS,
    n_mcmc_steps: int = DEFAULT_N_MCMC_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    resampling_fn: Callable = resampling.systematic,
    score_batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
    verbose: bool = True,
) -> Tuple[object, List[dict], List[float], np.ndarray]:
    """Run adaptive-tempered SMC until the tempering parameter reaches 1.

    Parameters
    ----------
    log_prior_fn, log_likelihood_fn : single-particle theta -> scalar, same
        contract as smc_tempered.run_tempered_smc. Only the likelihood is
        tempered; the prior holds at full strength throughout, so an
        informative prior (priors.restraint_prior over connectivity, say)
        keeps every particle physically plausible at lambda = 0.
    log_prob_fn : full (lambda = 1) log-posterior, for best-particle tracking.
    initial_particles : batched pytree from
        smc_particles.initialize_particles.
    proposal_fn : the symmetric SO(3)-aware kernel from proposals.py.
    target_ess : effective sample size to hold, as a fraction of the
        population, when solving for each temperature increment.
    max_steps : safety stop; reaching it is logged as a warning and the run
        returns whatever it has, with the final lambda short of 1.

    Returns
    -------
    final_state : blackjax.smc.tempered.TemperedSMCState
    best_thetas : best-scoring particle per step, including the initial one.
    best_scores : matching log_prob_fn values.
    lambdas : the ladder the sampler chose, starting at 0. Its length is a
        result of the run, not an input.
    """
    if not 0.0 < target_ess < 1.0:
        raise ValueError(f"target_ess must lie strictly between 0 and 1; got {target_ess}")

    n_particles = particle_count(initial_particles)
    mcmc_step_fn, mcmc_init_fn = build_rmh_mcmc_pair(proposal_fn)

    # Built and compiled once: the temperature is solved for inside the
    # kernel, so unlike a fixed ladder there is no per-step Python constant to
    # retrace against.
    kernel = jax.jit(
        adaptive_tempered.build_kernel(
            log_prior_fn,
            log_likelihood_fn,
            mcmc_step_fn,
            mcmc_init_fn,
            resampling_fn,
            target_ess,
        ),
        static_argnums=(2,),  # num_mcmc_steps sets a lax.scan length
    )

    best_particle = best_particle_selector(log_prob_fn, batch_size=score_batch_size)

    state = tempered.init(initial_particles)
    best_thetas: List[dict] = []
    best_scores: List[float] = []
    lambdas: List[float] = [0.0]
    theta0, score0 = best_particle(state.particles)
    best_thetas.append(jax.tree_util.tree_map(np.array, theta0))
    best_scores.append(score0)

    logger.info(
        "Running adaptive-tempered SMC: %d particles, target_ess=%.2f, %d MCMC "
        "sweeps/step, max_steps=%d",
        n_particles,
        target_ess,
        n_mcmc_steps,
        max_steps,
    )

    timer = start_timing()
    step_idx = 0
    while float(state.tempering_param) < 1.0 and step_idx < max_steps:
        step_idx += 1
        rng_key, step_key = jax.random.split(rng_key)
        state, info = kernel(step_key, state, n_mcmc_steps, {})

        lambdas.append(float(state.tempering_param))
        theta_i, score_i = best_particle(state.particles)
        best_thetas.append(jax.tree_util.tree_map(np.array, theta_i))
        best_scores.append(score_i)

        if verbose:
            logger.info(
                "adaptive SMC step %3d | lambda=%.4f (+%.4f) | best log-post=%.2f | "
                "mean accept=%.1f%%",
                step_idx,
                lambdas[-1],
                lambdas[-1] - lambdas[-2],
                score_i,
                100 * mean_acceptance_rate(info),
            )

    elapsed = elapsed_timing(timer)
    if step_idx <= 1 and float(state.tempering_param) >= 1.0:
        # Reaching lambda = 1 in one stride means reweighting never cost any
        # effective sample size, i.e. the log-likelihood is near-constant
        # across the population. Usually that means the restraints were
        # partitioned wrongly and the term that actually discriminates
        # between structures ended up in the (untempered) prior, leaving
        # nothing for the anneal to do.
        logger.warning(
            "adaptive-tempered SMC reached lambda=1 in a single step: the log-likelihood "
            "barely varies across the particle population, so tempering it is a no-op. "
            "Check that the discriminating restraints are in the likelihood scoring "
            "function rather than in the prior."
        )
    if float(state.tempering_param) < 1.0:
        logger.warning(
            "adaptive-tempered SMC stopped at max_steps=%d with lambda=%.4f < 1: the ESS "
            "solver is taking vanishing steps. Lower target_ess, widen the proposal "
            "scales, or use a more informative prior.",
            max_steps,
            float(state.tempering_param),
        )

    logger.info(
        "adaptive-tempered SMC finished: %d particles x %d adaptive steps (lambda=%.4f) in "
        "%.2fs wall / %.2fs cpu, best log-post=%.2f",
        n_particles,
        step_idx,
        float(state.tempering_param),
        elapsed.wall_time,
        elapsed.cpu_time,
        best_scores[-1],
    )

    return state, best_thetas, best_scores, np.asarray(lambdas)
