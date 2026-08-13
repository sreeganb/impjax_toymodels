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

import time
from typing import Callable, List, Tuple

import blackjax
import jax
import numpy as np


def run_custom_proposal_rmh(
    rng_key: jax.Array,
    log_prob_fn: Callable,
    initial_position,
    proposal_fn: Callable,
    n_steps: int = 1000,
    burnin: int = 0,
    thin: int = 1,
    verbose: bool = True,
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

    if verbose:
        print(f"Running custom-proposal RMH: {n_steps} steps")

    t0 = time.time()
    curr_state = state
    print_every = max(1, n_steps // 10)

    for i in range(n_steps):
        curr_state, info = step_fn(keys[i], curr_state)
        accepts.append(float(info.is_accepted))

        if i >= burnin and (i - burnin) % thin == 0:
            positions.append(jax.tree_util.tree_map(np.array, curr_state.position))
            log_probs.append(float(curr_state.logdensity))

        if verbose and (i + 1) % print_every == 0:
            recent_acc = np.mean(accepts[-min(1000, len(accepts)) :])
            print(
                f"  Step {i + 1:6d}/{n_steps} | LogProb: {curr_state.logdensity:10.2f} "
                f"| Accept: {recent_acc:.1%}"
            )

    dt = time.time() - t0
    overall_acc = float(np.mean(accepts))
    if verbose:
        print(f"Completed in {dt:.2f}s ({n_steps / dt:.0f} steps/s)")
        print(f"Overall acceptance rate: {overall_acc:.1%}")
        print(f"Saved {len(positions)} samples")

    return positions, np.array(log_probs), overall_acc
