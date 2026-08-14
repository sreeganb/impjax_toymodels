"""Drive BlackJAX's generic Metropolis kernel with a caller-supplied proposal.

`rmh_sampler.py` wraps `blackjax.mcmc.random_walk.normal_random_walk` /
`additive_step_random_walk`, which assume a flat array position and an
additive step. Our SO(3) rigid-body proposal (proposals.py) is not additive
(it composes quaternions) and the reduced state is a pytree, not a flat
array, so it needs BlackJAX's other, more general constructor instead:
`blackjax.rmh(logdensity_fn, proposal)`, which accepts any pytree position
and any *symmetric* `proposal(key, position) -> new_position` with no
Hastings correction. This module does not implement a sampling algorithm --
every step below is exactly `blackjax.rmh(...).step`, mirroring the driving
loop already used in `rmh_sampler.run_rmh_sampling`.
"""

import logging
from typing import Callable, List, Optional, Tuple

import blackjax
import jax
import numpy as np

from .timing import elapsed_timing, start_timing

logger = logging.getLogger(__name__)


def run_custom_proposal_rmh(
    rng_key: jax.Array,
    log_prob_fn: Callable,
    initial_position,
    proposal_fn: Callable,
    n_steps: int = 1000,
    burnin: int = 0,
    thin: int = 1,
    verbose: bool = True,
    step_callback: Optional[Callable[[int, dict, float, bool], None]] = None,
    step_callback_every: int = 1,
) -> Tuple[List, np.ndarray, float]:
    """Run Metropolis sampling on a pytree state with an arbitrary symmetric proposal.

    Parameters
    ----------
    rng_key : jax.Array
    log_prob_fn : Callable
        Log-density: position (pytree) -> scalar.
    initial_position : pytree
        Starting state, e.g. the {"quaternions", "translations", "bead_coords"}
        dict produced by state_sync.extract.
    proposal_fn : Callable
        `proposal_fn(key, position) -> new_position`. Must be symmetric (see
        proposals.py) -- this driver applies the plain Metropolis ratio with
        no Hastings correction term.
    n_steps, burnin, thin : int
        Standard MCMC loop controls, matching rmh_sampler.run_rmh_sampling.
    step_callback : optional callback, same shape as
        rmh_sampler.run_rmh_sampling's: `callback(step_index, position,
        log_prob, is_accepted)`. Used for DEBUG-mode JAX-vs-CPU-IMP score
        verification (score_verification.py) without this sampler-agnostic
        driver needing to know anything about IMP.
    step_callback_every : only invoke `step_callback` every this many steps
        (default every step, matching rmh_sampler's callback; pass a larger
        value for expensive callbacks such as a CPU IMP score check).

    Returns
    -------
    positions : list of pytrees, one saved sample per (post-burnin, thinned) step.
    log_probs : np.ndarray of log-densities at the saved samples.
    acceptance_rate : float
    """
    kernel = blackjax.rmh(log_prob_fn, proposal_fn)
    state = kernel.init(initial_position)
    step_fn = jax.jit(kernel.step)
    keys = jax.random.split(rng_key, n_steps)

    positions: List = []
    log_probs: List[float] = []
    accepts: List[float] = []

    logger.debug("Running custom-proposal RMH: %d steps", n_steps)

    timer = start_timing()
    curr_state = state
    print_every = max(1, n_steps // 10)

    for i in range(n_steps):
        curr_state, info = step_fn(keys[i], curr_state)
        accepts.append(float(info.is_accepted))

        if i >= burnin and (i - burnin) % thin == 0:
            positions.append(jax.tree_util.tree_map(np.array, curr_state.position))
            log_probs.append(float(curr_state.logdensity))

        if step_callback is not None and i % step_callback_every == 0:
            step_callback(i, curr_state.position, float(curr_state.logdensity), bool(info.is_accepted))

        if verbose and (i + 1) % print_every == 0:
            recent_acc = np.mean(accepts[-min(1000, len(accepts)) :])
            logger.info(
                "Step %6d/%d | LogProb: %10.2f | Accept: %.1f%%",
                i + 1,
                n_steps,
                curr_state.logdensity,
                100 * recent_acc,
            )

    elapsed = elapsed_timing(timer)
    overall_acc = float(np.mean(accepts))
    # Always logged (INFO), regardless of `verbose`: this is the run summary
    # that belongs in a log file even when per-step console progress is off.
    logger.info(
        "custom_rmh finished: %d steps in %.2fs wall / %.2fs cpu (%.0f steps/s), "
        "acceptance rate %.1f%%, %d samples saved",
        n_steps,
        elapsed.wall_time,
        elapsed.cpu_time,
        n_steps / elapsed.wall_time if elapsed.wall_time > 0 else float("inf"),
        100 * overall_acc,
        len(positions),
    )

    return positions, np.array(log_probs), overall_acc
