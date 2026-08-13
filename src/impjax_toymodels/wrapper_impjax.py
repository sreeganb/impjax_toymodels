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

from . import custom_rmh, dof_layout, gpu_io, logging_config, proposals, state_sync

logger = logging.getLogger(__name__)


@dataclass
class WrapperContext:
    """Everything build_log_prob assembles once from a BuiltSystem + scoring function."""

    layout: dof_layout.SystemLayout
    log_prob_fn: Callable[[dict], jnp.ndarray]
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

    def log_prob_fn(theta: dict) -> jnp.ndarray:
        xyz = expand(theta)
        neg_energy = -jax_interface.score_func({"xyz": xyz, "r": r})
        if prior_fn is None:
            return neg_energy
        return neg_energy + prior_fn(theta)

    initial_theta = jax.tree_util.tree_map(
        jnp.asarray, state_sync.extract(built_system, layout)
    )
    return WrapperContext(layout=layout, log_prob_fn=log_prob_fn, initial_theta=initial_theta)


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
        "sigma_translation=%s sigma_bead=%s burnin=%d thin=%d",
        n_steps,
        mode,
        sigma_rotation,
        sigma_translation,
        sigma_bead,
        burnin,
        thin,
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

    positions, log_probs, acceptance_rate = custom_rmh.run_custom_proposal_rmh(
        rng_key,
        context.log_prob_fn,
        context.initial_theta,
        proposal_fn,
        n_steps=n_steps,
        burnin=burnin,
        thin=thin,
        verbose=verbose,
    )

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
