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
    priors,
    proposals,
    score_verification,
    smc_adaptive_tempered,
    smc_fixed_schedule,
    smc_particles,
    smc_tempered,
    state_sync,
)

logger = logging.getLogger(__name__)

# The three SMC variants selectable through run_smc_sampling(variant=...).
SMC_VARIANTS = ("fixed", "tempered", "adaptive")


@dataclass
class WrapperContext:
    """Everything build_log_prob assembles once from a BuiltSystem + scoring function."""

    layout: dof_layout.SystemLayout
    log_prob_fn: Callable[[dict], jnp.ndarray]
    log_likelihood_fn: Callable[[dict], jnp.ndarray]
    log_prior_fn: Callable[[dict], jnp.ndarray]
    initial_theta: dict
    prior: priors.Prior


def build_log_prob(built_system, score_function, prior=None) -> WrapperContext:
    """Build the JAX log-posterior for a BuiltSystem (doc/design.tex Eq. 5).

    Parameters
    ----------
    built_system : system_info.BuiltSystem
    score_function : IMP.core.RestraintsScoringFunction
        Must already have been evaluated at least once so `_get_jax()` has a
        valid exported model. When a restraint-based prior is in use, this
        holds only the *likelihood* restraints -- the prior's restraints live
        in their own scoring function (see priors.restraint_prior).
    prior : None, a `priors.Prior`, or a `priors.PriorFactory`
        The prior p0(theta). Defaults to `priors.flat()` (log p0 = 0), which
        is what this package did before priors became selectable. Choosing an
        informative prior matters most for SMC, whose lambda = 0 distribution
        *is* the prior -- see priors.py.

    Returns
    -------
    A WrapperContext exposing the three densities the samplers need kept
    apart: `log_likelihood_fn` (bare -score, the only term SMC tempers),
    `log_prior_fn` (untempered), and `log_prob_fn` (their sum, the full
    posterior, which is what plain RMH samples).
    """
    layout = dof_layout.build(built_system)
    jax_interface = score_function._get_jax()
    template_xyz, r = state_sync.capture_template(jax_interface)
    expand = state_sync.make_expansion_fn(layout, template_xyz)
    r = jnp.asarray(r)

    initial_theta = jax.tree_util.tree_map(
        jnp.asarray, state_sync.extract(built_system, layout)
    )
    resolved_prior = priors.resolve(
        prior,
        priors.PriorContext(
            layout=layout,
            expand=expand,
            radii=r,
            initial_theta=initial_theta,
            score_function=score_function,
        ),
    )

    def log_likelihood_fn(theta: dict) -> jnp.ndarray:
        """The bare IMP-derived term, -score(theta), with no prior mixed in --
        this is what SMC's tempering formula (log_prior + lambda * log_likelihood)
        needs; run_sampling's plain RMH only ever uses log_prob_fn below."""
        xyz = expand(theta)
        return -jax_interface.score_func({"xyz": xyz, "r": r})

    def log_prob_fn(theta: dict) -> jnp.ndarray:
        return log_likelihood_fn(theta) + resolved_prior.log_prob(theta)

    return WrapperContext(
        layout=layout,
        log_prob_fn=log_prob_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_prior_fn=resolved_prior.log_prob,
        initial_theta=initial_theta,
        prior=resolved_prior,
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
    prior=None,
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
    prior : the prior p0(theta); see build_log_prob and priors.py. RMH
        samples the full posterior, so the prior simply adds to the target
        here (unlike SMC, where it also defines the lambda = 0 distribution).
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

    context = build_log_prob(built_system, score_function, prior=prior)
    run_logger.info("prior: %s", context.prior.name)
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


def run_smc_sampling(
    built_system,
    score_function,
    rng_key: jax.Array,
    variant: str = "fixed",
    n_particles: int = smc_fixed_schedule.DEFAULT_N_PARTICLES,
    n_temperature_steps: int = smc_fixed_schedule.DEFAULT_N_TEMPERATURE_STEPS,
    schedule: str = smc_fixed_schedule.DEFAULT_SCHEDULE,
    n_mcmc_steps: int = smc_fixed_schedule.DEFAULT_N_MCMC_STEPS,
    target_ess: float = smc_adaptive_tempered.DEFAULT_TARGET_ESS,
    max_steps: int = smc_adaptive_tempered.DEFAULT_MAX_STEPS,
    mode: str = "all",
    sigma_rotation: float = 0.05,
    sigma_translation: float = 1.0,
    sigma_bead: float = 1.0,
    prior=None,
    sync_back: bool = True,
    verbose: bool = True,
    rmf_path: Optional[str] = None,
    stat_path: Optional[str] = None,
    log_path: Optional[str] = None,
    debug: bool = False,
    debug_every: int = 1,
    score_comparison_path: Optional[str] = None,
) -> Tuple[List[dict], List[float], np.ndarray]:
    """Run SMC over a BuiltSystem -- the SMC counterpart to run_sampling, same
    end-to-end contract (build, sample, optionally write RMF3/stat/log/debug
    output in one call).

    Only the best-scoring particle *per temperature step* is ever written out
    (as one RMF3 frame each, forming a trajectory of the best model's
    progression through the anneal) -- the other `n_particles - 1` particles
    at each step are never persisted.

    Parameters
    ----------
    variant : which of the three SMC samplers to drive, all BlackJAX's:
        "fixed"     smc_fixed_schedule.py over `blackjax.smc.base`: a
                    user-set temperature ladder, mutation closures rebuilt
                    (and retraced) per step. The original, slowest variant.
        "tempered"  smc_tempered.py over `blackjax.smc.tempered`: the same
                    user-set ladder, but the kernel is built and JIT-compiled
                    once with the temperature as a traced argument.
        "adaptive"  smc_adaptive_tempered.py over
                    `blackjax.smc.adaptive_tempered`: the ladder is solved
                    for at each step to hold the effective sample size at
                    `target_ess`, so the run takes the temperature steps it
                    needs rather than a fixed count.
    n_temperature_steps, schedule : the ladder, for "fixed" and "tempered";
        ignored by "adaptive", which chooses its own.
    target_ess, max_steps : for "adaptive" only; ignored by the other two.
    prior : the prior p0(theta); see build_log_prob and priors.py. This
        matters more here than in run_sampling: SMC's lambda = 0
        distribution *is* the prior, so an informative one (connectivity via
        priors.restraint_prior, say) both seeds the population with
        physically plausible structures and holds them plausible through the
        whole anneal. When the prior supplies a sampler, the initial
        population is drawn from it; otherwise it is built by perturbing the
        model as built (see smc_particles.initialize_particles).
    debug_every : "every debug_every temperature steps" (default: every step,
        since there are typically only tens of them).

    Other parameters mirror run_sampling's (mode, sigmas, sync_back,
    rmf_path/stat_path/log_path, debug/score_comparison_path).

    Returns
    -------
    best_thetas : list of theta dicts, one per temperature step (including
        the initial one) -- the best-scoring particle at that step. For
        "adaptive" its length is a result of the run, not an input.
    best_scores : matching list of log-posterior values.
    lambdas : the temperature ladder actually walked.
    """
    if variant not in SMC_VARIANTS:
        raise ValueError(f"Unknown SMC variant {variant!r}; choose from {SMC_VARIANTS}")

    run_logger = logging_config.configure_logging(log_path=log_path)
    run_logger.info(
        "run_smc_sampling starting: variant=%s n_particles=%d n_temperature_steps=%d "
        "schedule=%s n_mcmc_steps=%d target_ess=%.2f mode=%s debug=%s",
        variant,
        n_particles,
        n_temperature_steps,
        schedule,
        n_mcmc_steps,
        target_ess,
        mode,
        debug,
    )

    context = build_log_prob(built_system, score_function, prior=prior)
    run_logger.info("prior: %s", context.prior.name)
    proposal_fn = proposals.build_composite(
        context.layout, sigma_rotation, sigma_translation, sigma_bead, mode=mode
    )

    init_key, run_key = jax.random.split(rng_key)
    initial_particles = smc_particles.initialize_particles(
        init_key, n_particles, context.initial_theta, context.prior, proposal_fn
    )

    # Every variant takes the same four densities/populations; only the
    # schedule-control arguments differ. `log_likelihood_fn` is the bare
    # -score term -- passing log_prob_fn here would temper the prior too and
    # double-count it at lambda = 1.
    common = (
        run_key,
        context.log_prior_fn,
        context.log_likelihood_fn,
        context.log_prob_fn,  # full posterior, used only for best-particle reporting
        initial_particles,
        proposal_fn,
    )
    if variant == "fixed":
        state, best_thetas, best_scores, lambdas = smc_fixed_schedule.run_fixed_schedule_smc(
            *common,
            n_temperature_steps=n_temperature_steps,
            schedule=schedule,
            n_mcmc_steps=n_mcmc_steps,
            verbose=verbose,
        )
    elif variant == "tempered":
        state, best_thetas, best_scores, lambdas = smc_tempered.run_tempered_smc(
            *common,
            n_temperature_steps=n_temperature_steps,
            schedule=schedule,
            n_mcmc_steps=n_mcmc_steps,
            verbose=verbose,
        )
    else:
        state, best_thetas, best_scores, lambdas = smc_adaptive_tempered.run_adaptive_tempered_smc(
            *common,
            target_ess=target_ess,
            max_steps=max_steps,
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
        "run_smc_sampling finished: variant=%s, %d temperature steps, best log-post=%.2f%s",
        variant,
        len(lambdas) - 1,
        best_scores[-1],
        f", rmf={rmf_path}" if rmf_path else "",
    )

    return best_thetas, best_scores, lambdas
