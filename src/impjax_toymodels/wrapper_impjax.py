"""The single pipeline entry point: IMP model -> JAX log-density -> BlackJAX
sampling -> synced-back IMP model.

Per planning.md, this is meant to be the only file a user's own setup script
touches: it ties together dof_layout.py (index bookkeeping), state_sync.py
(the IMP<->JAX bridge, doc/design.tex Section 3-4), proposals.py (SO(3)-
compliant moves, Section 5) and custom_rmh.py (the BlackJAX driving loop,
Section 6/Algorithm 1) -- no sampling algorithm or scoring function is
implemented here, only inputs assembled for BlackJAX and IMP's own JAX
export.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import (
    custom_rmh,
    dof_layout,
    gpu_io,
    logging_config,
    proposals,
    score_verification,
    smc_fixed_schedule,
    state_sync,
)

logger = logging.getLogger(__name__)


@dataclass
class WrapperContext:
    """Everything build_log_prob assembles once from a BuiltSystem + scoring function."""

    layout: dof_layout.SystemLayout
    log_prob_fn: Callable[[dict], jnp.ndarray]
    log_likelihood_fn: Callable[[dict], jnp.ndarray]
    initial_theta: dict


def build_log_prob(
    built_system, score_function, prior_fn: Optional[Callable[[dict], jnp.ndarray]] = None
) -> WrapperContext:
    """Build the JAX log-posterior for a BuiltSystem (doc/design.tex Eq. 5).

    Parameters
    ----------
    built_system : system_info.BuiltSystem
    score_function : IMP.core.RestraintsScoringFunction
        Must already have been evaluated at least once so `_get_jax()` has a
        valid exported model.
    prior_fn : Callable, optional
        log p0(theta); defaults to flat (0) if omitted.
    """
    layout = dof_layout.build(built_system)
    jax_interface = score_function._get_jax()
    template_xyz, r = state_sync.capture_template(jax_interface)
    expand = state_sync.make_expansion_fn(layout, template_xyz)
    r = jnp.asarray(r)

    def log_likelihood_fn(theta: dict) -> jnp.ndarray:
        """The bare IMP-derived term, -score(theta), with no prior mixed in --
        this is what SMC's tempering formula (log_prior + lambda * log_likelihood)
        needs; run_sampling's plain RMH only ever uses log_prob_fn below."""
        xyz = expand(theta)
        return -jax_interface.score_func({"xyz": xyz, "r": r})

    def log_prob_fn(theta: dict) -> jnp.ndarray:
        neg_energy = log_likelihood_fn(theta)
        if prior_fn is None:
            return neg_energy
        return neg_energy + prior_fn(theta)

    initial_theta = jax.tree_util.tree_map(
        jnp.asarray, state_sync.extract(built_system, layout)
    )
    return WrapperContext(
        layout=layout,
        log_prob_fn=log_prob_fn,
        log_likelihood_fn=log_likelihood_fn,
        initial_theta=initial_theta,
    )


def run_sampling(
    built_system,
    score_function,
    rng_key: jax.Array,
    n_steps: int = 1000,
    mode: str = "all",
    sigma_rotation: float = 0.05,
    sigma_translation: float = 1.0,
    sigma_bead: float = 1.0,
    prior_fn: Optional[Callable[[dict], jnp.ndarray]] = None,
    burnin: int = 0,
    thin: int = 1,
    sync_back: bool = True,
    verbose: bool = True,
    rmf_path: Optional[str] = None,
    stat_path: Optional[str] = None,
    log_path: Optional[str] = None,
    debug: bool = False,
    debug_every: int = 50,
    score_comparison_path: Optional[str] = None,
) -> Tuple[List[dict], np.ndarray, float]:
    """Run RMH sampling over a BuiltSystem's rigid bodies/beads -- the full
    pipeline: build the log-posterior and proposal, run BlackJAX, and
    (optionally) write out an RMF3 trajectory, a stat file, and a run log.

    This is the single call planning.md asks the wrapper to be: a user's own
    setup script builds a `BuiltSystem` and an
    `IMP.core.RestraintsScoringFunction`, then calls this.

    Parameters
    ----------
    mode : one of the five planning.md sampling modes (see
        dof_layout.SAMPLING_MODES): "rotation", "translation", "rigid",
        "beads", or "all".
    rmf_path : if given, every saved sample is written as an RMF3 frame
        (doc/design.tex Section 7 / gpu_io.py), all at once at the end of
        this call -- true incremental, on-device-buffered flushing every K
        steps is a tracked follow-up, not yet implemented here.
    stat_path : stat-file path for the same run; defaults to
        `rmf_path` with a "_stats.csv" suffix if omitted.
    log_path : if given, this run's log messages (always including a final
        acceptance-rate/timing summary, regardless of `verbose`) are also
        written here, on top of the console.
    sync_back : only meaningful when `rmf_path` is not given. Writing RMF3
        frames necessarily stages each state into the live IMP model (that
        is how IMP's RMF writer works), so when `rmf_path` is given the
        model ends up holding the last written sample's state regardless of
        `sync_back`.
    debug : if set, every `debug_every` steps this recomputes the score at
        the current configuration using IMP's own CPU implementation
        (ground truth) and compares it against the JAX score already used
        for sampling, logging a warning on any mismatch and recording every
        checkpoint to `score_comparison_path` (see score_verification.py).
        Off by default and checked infrequently when on, since a CPU
        `evaluate()` call is comparatively expensive.
    score_comparison_path : CSV path for the debug score comparison;
        required if `debug` is set and neither this nor `log_path` is given
        (defaults to `log_path` with a "_score_comparison.csv" suffix).

    Returns
    -------
    positions : list of {"quaternions", "translations", "bead_coords"} dicts,
        one per saved (post-burnin, thinned) step.
    log_probs : np.ndarray of log-posterior values at the saved samples.
    acceptance_rate : float
    """
    run_logger = logging_config.configure_logging(log_path=log_path)
    run_logger.info(
        "run_sampling starting: n_steps=%d mode=%s sigma_rotation=%s "
        "sigma_translation=%s sigma_bead=%s burnin=%d thin=%d debug=%s",
        n_steps,
        mode,
        sigma_rotation,
        sigma_translation,
        sigma_bead,
        burnin,
        thin,
        debug,
    )

    context = build_log_prob(built_system, score_function, prior_fn=prior_fn)
    run_logger.debug(
        "Built log-posterior: %d rigid bodies, %d beads (flat_size=%d)",
        context.layout.n_rigid_bodies,
        context.layout.n_beads,
        context.layout.flat_size,
    )
    proposal_fn = proposals.build_composite(
        context.layout, sigma_rotation, sigma_translation, sigma_bead, mode=mode
    )

    score_writer = None
    step_callback = None
    if debug:
        resolved_comparison_path = score_comparison_path
        if resolved_comparison_path is None:
            if log_path is not None:
                resolved_comparison_path = f"{os.path.splitext(log_path)[0]}_score_comparison.csv"
            else:
                raise ValueError(
                    "debug=True needs score_comparison_path (or log_path, to derive it from)"
                )
        score_writer = score_verification.ScoreComparisonWriter(resolved_comparison_path)
        step_callback = score_verification.make_debug_callback(
            score_writer, context.layout, built_system, score_function, debug_every
        )

    try:
        positions, log_probs, acceptance_rate = custom_rmh.run_custom_proposal_rmh(
            rng_key,
            context.log_prob_fn,
            context.initial_theta,
            proposal_fn,
            n_steps=n_steps,
            burnin=burnin,
            thin=thin,
            verbose=verbose,
            step_callback=step_callback,
            step_callback_every=1,  # make_debug_callback already applies debug_every itself
        )
    finally:
        if score_writer is not None:
            score_writer.close()

    if rmf_path is not None:
        resolved_stat_path = stat_path or f"{os.path.splitext(rmf_path)[0]}_stats.csv"
        with gpu_io.TrajectoryWriter(rmf_path, resolved_stat_path, built_system.root_hier) as writer:
            gpu_io.write_block(writer, positions, log_probs, context.layout, built_system)
    elif sync_back and positions:
        state_sync.apply(positions[-1], context.layout, built_system)

    run_logger.info(
        "run_sampling finished: acceptance_rate=%.1f%% samples_saved=%d%s",
        100 * acceptance_rate,
        len(positions),
        f", rmf={rmf_path}" if rmf_path else "",
    )

    return positions, log_probs, acceptance_rate


def _replicate_and_spread(theta: dict, n_particles: int, key: jax.Array, proposal_fn) -> dict:
    """Build an initial SMC particle population: `n_particles` copies of
    `theta`, each independently nudged once by `proposal_fn` so they don't
    start out identical (a flat/near-flat prior gives no other way to spread
    an initial population -- see doc/design.tex's SMC section)."""
    replicated = jax.tree_util.tree_map(
        lambda leaf: jnp.tile(leaf[None, ...], (n_particles,) + (1,) * leaf.ndim), theta
    )
    keys = jax.random.split(key, n_particles)
    return jax.vmap(proposal_fn)(keys, replicated)


def run_smc_sampling(
    built_system,
    score_function,
    rng_key: jax.Array,
    n_particles: int = smc_fixed_schedule.DEFAULT_N_PARTICLES,
    n_temperature_steps: int = smc_fixed_schedule.DEFAULT_N_TEMPERATURE_STEPS,
    schedule: str = smc_fixed_schedule.DEFAULT_SCHEDULE,
    n_mcmc_steps: int = smc_fixed_schedule.DEFAULT_N_MCMC_STEPS,
    mode: str = "all",
    sigma_rotation: float = 0.05,
    sigma_translation: float = 1.0,
    sigma_bead: float = 1.0,
    prior_fn: Optional[Callable[[dict], jnp.ndarray]] = None,
    sync_back: bool = True,
    verbose: bool = True,
    rmf_path: Optional[str] = None,
    stat_path: Optional[str] = None,
    log_path: Optional[str] = None,
    debug: bool = False,
    debug_every: int = 1,
    score_comparison_path: Optional[str] = None,
) -> Tuple[List[dict], List[float], np.ndarray]:
    """Run fixed-schedule (untuned) SMC over a BuiltSystem -- the SMC
    counterpart to run_sampling, same end-to-end contract (build, sample,
    optionally write RMF3/stat/log/debug output in one call).

    Only the best-scoring particle *per temperature step* is ever written
    out (as one RMF3 frame each, forming a trajectory of the best model's
    progression through the anneal) -- the other `n_particles - 1` particles
    at each step are never persisted. See smc_fixed_schedule.py for the
    fixed (non-adaptive) tempering/mutation/resampling mechanics and
    doc/design.tex's SMC section for defaults and rationale.

    Parameters mirror run_sampling's where they overlap (mode, sigmas,
    prior_fn, sync_back, rmf_path/stat_path/log_path, debug/
    score_comparison_path); n_particles/n_temperature_steps/schedule/
    n_mcmc_steps are SMC-specific, defaulting to smc_fixed_schedule.py's
    untuned constants. `debug_every` here means "every debug_every
    temperature steps" (default: every step, since there are typically only
    ~20 of them, far fewer than an RMH run's steps).

    Returns
    -------
    best_thetas : list of theta dicts, one per temperature step (including
        the initial one) -- the best-scoring particle at that step.
    best_scores : matching list of log-posterior values.
    lambdas : the fixed temperature schedule used.
    """
    run_logger = logging_config.configure_logging(log_path=log_path)
    run_logger.info(
        "run_smc_sampling starting: n_particles=%d n_temperature_steps=%d "
        "schedule=%s n_mcmc_steps=%d mode=%s debug=%s",
        n_particles,
        n_temperature_steps,
        schedule,
        n_mcmc_steps,
        mode,
        debug,
    )

    context = build_log_prob(built_system, score_function, prior_fn=prior_fn)
    proposal_fn = proposals.build_composite(
        context.layout, sigma_rotation, sigma_translation, sigma_bead, mode=mode
    )

    spread_key, run_key = jax.random.split(rng_key)
    initial_particles = _replicate_and_spread(context.initial_theta, n_particles, spread_key, proposal_fn)

    def log_prior_fn(theta: dict) -> jnp.ndarray:
        return jnp.asarray(0.0) if prior_fn is None else prior_fn(theta)

    state, best_thetas, best_scores, lambdas = smc_fixed_schedule.run_fixed_schedule_smc(
        run_key,
        log_prior_fn,
        context.log_likelihood_fn,  # bare -score term; log_prob_fn would double-count prior_fn
        context.log_prob_fn,  # full posterior, used only for best-particle reporting
        initial_particles,
        proposal_fn,
        n_temperature_steps=n_temperature_steps,
        schedule=schedule,
        n_mcmc_steps=n_mcmc_steps,
        verbose=verbose,
    )

    if debug:
        resolved_comparison_path = score_comparison_path
        if resolved_comparison_path is None:
            if log_path is None:
                raise ValueError(
                    "debug=True needs score_comparison_path (or log_path, to derive it from)"
                )
            resolved_comparison_path = f"{os.path.splitext(log_path)[0]}_score_comparison.csv"
        with score_verification.ScoreComparisonWriter(resolved_comparison_path) as score_writer:
            for step, (theta, score) in enumerate(zip(best_thetas, best_scores)):
                if step % debug_every == 0:
                    score_writer.record(step, theta, context.layout, built_system, score_function, score)

    if rmf_path is not None:
        resolved_stat_path = stat_path or f"{os.path.splitext(rmf_path)[0]}_stats.csv"
        with gpu_io.TrajectoryWriter(rmf_path, resolved_stat_path, built_system.root_hier) as writer:
            gpu_io.write_block(writer, best_thetas, best_scores, context.layout, built_system)
    elif sync_back and best_thetas:
        state_sync.apply(best_thetas[-1], context.layout, built_system)

    run_logger.info(
        "run_smc_sampling finished: %d temperature steps, best log-post=%.2f%s",
        n_temperature_steps,
        best_scores[-1],
        f", rmf={rmf_path}" if rmf_path else "",
    )

    return best_thetas, best_scores, lambdas
