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

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import custom_rmh, dof_layout, proposals, state_sync


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
) -> Tuple[List[dict], np.ndarray, float]:
    """Run RMH sampling over a BuiltSystem's rigid bodies/beads.

    `mode` selects one of the five planning.md sampling modes (see
    dof_layout.SAMPLING_MODES): "rotation", "translation", "rigid", "beads",
    or "all". If `sync_back` is set, the last saved sample is written back
    into the live IMP model (via state_sync.apply) so it can be inspected or
    written out with IMP's own RMF3/stat-file tools (see gpu_io.py for a
    batched, GPU-aware version of that sync-and-write step).

    Returns
    -------
    positions : list of {"quaternions", "translations", "bead_coords"} dicts,
        one per saved (post-burnin, thinned) step.
    log_probs : np.ndarray of log-posterior values at the saved samples.
    acceptance_rate : float
    """
    context = build_log_prob(built_system, score_function, prior_fn=prior_fn)
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

    if sync_back and positions:
        state_sync.apply(positions[-1], context.layout, built_system)

    return positions, log_probs, acceptance_rate
