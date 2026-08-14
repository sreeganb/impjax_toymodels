"""Particle-population helpers shared by every SMC variant in this package.

All three SMC runners (smc_fixed_schedule.py over `blackjax.smc.base`,
smc_tempered.py and smc_adaptive_tempered.py over `blackjax.smc.tempered`)
carry their particles as a *batched pytree* -- our reduced state
{"quaternions", "translations", "bead_coords"} with an extra leading
`n_particles` axis on every leaf -- rather than as the flat
`(n_particles, n_dims)` array `smc_base_sampler.py` assumes. BlackJAX itself
is pytree-native throughout, so nothing here re-implements sampling; these
are only the bookkeeping operations BlackJAX does not provide: counting
particles, pulling one out, scoring a whole population within a memory
budget, and building the initial population.

Kept in its own file so the three runners share one implementation instead of
three copies, and so a runner can be deleted without taking the helpers with
it (planning.md's modularity rule).
"""

import logging
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from .priors import Prior

logger = logging.getLogger(__name__)

# Default chunk size when scoring a population; bounds peak device memory at
# O(batch_size) evaluations of the (potentially large) expansion + score.
DEFAULT_SCORE_BATCH_SIZE = 16


def particle_count(particles) -> int:
    """Number of particles, read off the leading axis of any pytree leaf."""
    return jax.tree_util.tree_leaves(particles)[0].shape[0]


def select_particle(particles, index: int):
    """Pull a single (unbatched) particle out of a batched pytree population."""
    return jax.tree_util.tree_map(lambda leaf: leaf[index], particles)


def batched_scorer(
    fn: Callable[[dict], jnp.ndarray], batch_size: int = DEFAULT_SCORE_BATCH_SIZE
) -> Callable:
    """Apply `fn` (single particle -> scalar) across a batched pytree in chunks.

    Equivalent to `jax.vmap(fn)(particles)` but evaluates at most
    `batch_size` particles at a time, so peak memory is O(batch_size) rather
    than O(n_particles). This is the pytree-native counterpart of
    smc_base_sampler.batched_vmap, which indexes `inputs.shape[0]` directly
    and therefore only works on a flat array.
    """
    score_chunk = jax.jit(lambda chunk: jax.vmap(fn)(chunk))

    def scored(particles) -> jnp.ndarray:
        n = particle_count(particles)
        if n <= batch_size:
            return score_chunk(particles)
        results = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = jax.tree_util.tree_map(lambda leaf: leaf[start:end], particles)
            chunk_result = score_chunk(chunk)
            jax.block_until_ready(chunk_result)
            results.append(chunk_result)
        return jnp.concatenate(results, axis=0)

    return scored


def best_particle_selector(
    log_prob_fn: Callable[[dict], jnp.ndarray],
    batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
) -> Callable[[dict], Tuple[dict, float]]:
    """Build `particles -> (best particle, its log-posterior)`.

    Every SMC runner reports and writes out only the best-scoring particle
    per step (mirroring IMP's `ReplicaExchange(number_of_best_scoring_models
    =1)`), so the selection is factored out here once.
    """
    scorer = batched_scorer(log_prob_fn, batch_size=batch_size)

    def best(particles) -> Tuple[dict, float]:
        scores = scorer(particles)
        index = int(jnp.argmax(scores))
        return select_particle(particles, index), float(scores[index])

    return best


def initialize_particles(
    key: jax.Array,
    n_particles: int,
    initial_theta: dict,
    prior: Optional[Prior] = None,
    proposal_fn: Optional[Callable[[jax.Array, dict], dict]] = None,
) -> dict:
    """Build the SMC initial particle population.

    SMC's lambda = 0 distribution is the prior, so the statistically correct
    initialization is an i.i.d. draw from it -- taken here whenever the prior
    supplies a sampler (see priors.py). Priors that cannot be drawn from (an
    IMP restraint term, for instance) fall back to the older behaviour:
    replicate the built model's state and nudge each copy once with the
    proposal kernel, which at least spreads the population instead of leaving
    `n_particles` identical structures whose weights are degenerate. That
    fallback is a genuine approximation, not a draw from p0, and is logged as
    such.

    Parameters
    ----------
    n_particles : population size.
    initial_theta : the reduced state read out of the built IMP model, used
        by the fallback path.
    prior : resolved prior; its `sample` is used when present.
    proposal_fn : the SO(3)-aware kernel from proposals.py, required only for
        the fallback path.
    """
    if prior is not None and prior.sample is not None:
        logger.info("Initializing %d particles by sampling the prior (%s)", n_particles, prior.name)
        return jax.vmap(prior.sample)(jax.random.split(key, n_particles))

    if proposal_fn is None:
        raise ValueError(
            "initialize_particles needs either a prior with a sampler or a proposal_fn "
            "to spread replicated copies of initial_theta"
        )

    logger.info(
        "Initializing %d particles by perturbing the built model (prior %s has no sampler)",
        n_particles,
        "None" if prior is None else prior.name,
    )
    replicated = jax.tree_util.tree_map(
        lambda leaf: jnp.tile(leaf[None, ...], (n_particles,) + (1,) * leaf.ndim), initial_theta
    )
    return jax.vmap(proposal_fn)(jax.random.split(key, n_particles), replicated)
