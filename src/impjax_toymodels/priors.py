"""User-selectable prior distributions p0(theta) over the reduced sampling state.

Every sampler in this package factors its target as

    log pi_lambda(theta) = log p0(theta) + lambda * log L(theta)

where `log L` is the IMP-derived data/likelihood term (`-score`, built in
wrapper_impjax.build_log_prob) and `log p0` is whatever this module supplies.
Only the likelihood is tempered; the prior is enforced at full strength at
*every* lambda, including lambda = 0. That is precisely why the choice of
prior matters so much for SMC: the lambda = 0 distribution IS the prior, so a
flat prior makes the initial particle population an unstructured cloud of
disconnected coordinates that the anneal then has to repair. Handing the
structural knowledge we already have (a protein's residues are connected in
sequence) to the prior instead means every particle starts, and stays,
physically plausible -- see doc/design.tex's SMC section.

Three things a prior must do, captured by `Prior`: score a state
(`log_prob(theta) -> scalar`, JIT-traceable); optionally be drawn from
(`sample(key) -> theta`, used to initialize an SMC particle population *from
the prior*, as the SMC formalism calls for -- priors that cannot be sampled,
such as an IMP restraint term, leave this None and the caller falls back to
perturbing the current model state); and name itself, for the run log.

Priors are built through *factories* rather than constructed directly,
because some (`restraint_prior`) need machinery the wrapper only assembles
once it has read the IMP model: the expansion map Phi and the static radii
array. A factory is a `Callable[[PriorContext], Prior]`; the wrapper builds
the `PriorContext` and calls it. This keeps the module strictly
system-agnostic (per planning.md, `src/` never builds a specific IMP system).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .dof_layout import SystemLayout

# A single-particle state theta -> scalar log-density.
LogProbFn = Callable[[dict], jnp.ndarray]
# A PRNG key -> a single (unbatched) theta drawn from the prior.
SampleFn = Callable[[jax.Array], dict]


@dataclass(frozen=True)
class PriorContext:
    """Everything a prior factory may need, assembled once by the wrapper.

    Attributes
    ----------
    layout : the run's fixed index bookkeeping -- shape information for
        building a sampler.
    expand : the map Phi from theta to the full per-particle (N, 3)
        coordinate array (state_sync.make_expansion_fn); restraint-based
        priors need it to score a state the way IMP does.
    radii : static per-particle radii, which IMP's `score_func` expects
        alongside "xyz".
    initial_theta : the reduced state read out of the IMP model at build
        time, used as a reference point (e.g. to centre a bounding box on the
        system as built rather than on an arbitrary origin).
    score_function : the *likelihood* scoring function, kept only so
        `restraint_prior` can check the prior's restraints are not also
        scored as likelihood.
    """

    layout: SystemLayout
    expand: Callable[[dict], jnp.ndarray]
    radii: jnp.ndarray
    initial_theta: dict
    score_function: object


@dataclass(frozen=True)
class Prior:
    """A prior distribution over the reduced state, ready for a sampler."""

    name: str
    log_prob: LogProbFn
    sample: Optional[SampleFn] = None


# A factory defers construction until the wrapper can supply a PriorContext.
PriorFactory = Callable[[PriorContext], Prior]


def flat() -> PriorFactory:
    """The improper flat prior, log p0(theta) = 0 everywhere.

    This package's historical default: it contributes nothing to the target,
    so the posterior is the bare IMP score and SMC's lambda = 0 distribution
    is unbounded. No sampler, so SMC initialization falls back to perturbing
    the built model. Kept as an explicit, named choice so a run log records
    "flat" rather than silently recording nothing.
    """

    def factory(context: PriorContext) -> Prior:
        del context  # a flat prior needs nothing from the model
        return Prior(name="flat", log_prob=lambda theta: jnp.asarray(0.0), sample=None)

    return factory


def _uniform_quaternions(key: jax.Array, n: int) -> jnp.ndarray:
    """Draw `n` quaternions uniformly (Haar) from SO(3).

    Normalizing an isotropic 4-D Gaussian gives a uniform point on S^3, which
    is exactly the Haar measure on SO(3) under the usual double cover -- the
    SO(3)-correctness requirement proposals.py's rotation kernel obeys,
    applied to sampling rather than proposing.
    """
    raw = jax.random.normal(key, (n, 4))
    return raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)


def bounding_box(
    half_width: float = 100.0,
    center: Optional[Sequence[float]] = None,
    wall_sigma: float = 1.0,
) -> PriorFactory:
    """A soft axis-aligned bounding box on all translational degrees of freedom.

    The prior the package previously applied only implicitly (an unbounded
    flat prior is not normalizable; a box is what was meant). Making it
    explicit buys two things: the walls actually push particles back in, and
    -- more importantly for SMC -- the box *can* be sampled from, so an
    initial population is a genuine draw from p0 rather than a cloud of
    nudged copies of one structure. Rotations are left uniform on SO(3),
    contributing a constant to `log_prob` (dropped, since the density is only
    needed up to a constant) but honoured by `sample`.

    Parameters
    ----------
    half_width : half-side of the cube, in the model's length units
        (angstroms for IMP), applied per coordinate axis.
    center : box centre; defaults to the centroid of the built model's own
        sampled coordinates, so the box brackets the system as built rather
        than an arbitrary origin.
    wall_sigma : softness of the wall. A coordinate `d` units outside the box
        costs `0.5 * (d / wall_sigma)^2` -- harmonic beyond the faces, exactly
        flat inside, so the prior is a smooth (JIT- and gradient-friendly)
        stand-in for a hard uniform box.
    """

    def factory(context: PriorContext) -> Prior:
        layout = context.layout
        if center is None:
            # Centre on the built model: the mean of everything being sampled.
            movable = [np.asarray(context.initial_theta[k]).reshape(-1, 3)
                       for k in ("translations", "bead_coords")]
            box_center = jnp.asarray(
                np.concatenate([m for m in movable if m.size], axis=0).mean(axis=0)
            )
        else:
            box_center = jnp.asarray(center, dtype=jnp.float32)

        def _wall_penalty(coords: jnp.ndarray) -> jnp.ndarray:
            """Harmonic cost for however far each coordinate lies outside the box."""
            overshoot = jnp.maximum(jnp.abs(coords - box_center) - half_width, 0.0)
            return 0.5 * jnp.sum((overshoot / wall_sigma) ** 2)

        def log_prob(theta: dict) -> jnp.ndarray:
            return -(_wall_penalty(theta["translations"]) + _wall_penalty(theta["bead_coords"]))

        def sample(key: jax.Array) -> dict:
            key_q, key_t, key_b = jax.random.split(key, 3)
            return {
                "quaternions": _uniform_quaternions(key_q, layout.n_rigid_bodies),
                "translations": box_center
                + jax.random.uniform(
                    key_t, (layout.n_rigid_bodies, 3), minval=-half_width, maxval=half_width
                ),
                "bead_coords": box_center
                + jax.random.uniform(
                    key_b, (layout.n_beads, 3), minval=-half_width, maxval=half_width
                ),
            }

        return Prior(name=f"bounding_box(half_width={half_width})", log_prob=log_prob, sample=sample)

    return factory


def _restraint_names(score_function) -> Tuple[str, ...]:
    """Names of the restraints an IMP scoring function actually scores."""
    return tuple(restraint.get_name() for restraint in score_function.get_restraints())


def restraint_prior(prior_score_function, name: Optional[str] = None) -> PriorFactory:
    """Use a subset of the system's own IMP restraints as the prior.

    The structurally informed prior the flat/box priors cannot express: hand
    it a scoring function holding, say, the connectivity restraints (and
    optionally excluded volume), keep the data-derived restraints
    (crosslinks, EM, SAXS) in the scoring function passed to the wrapper as
    the likelihood, and SMC holds the chain connected at every temperature
    while tempering only the experimental data. Far less wasteful than a flat
    prior: no sampling effort goes into configurations that violate the
    protein's own sequence connectivity.

    Both scoring functions are evaluated through IMP's JAX export against the
    *same* per-particle coordinate array -- `get_jax_model()` is built from
    `Model.get_spheres_numpy()`, which is model-wide and indexed by particle
    index regardless of which restraints a scoring function holds (verified
    empirically against IMP develop-c881750dcd). So one expansion map Phi
    feeds both terms and no second index bookkeeping is needed.

    Parameters
    ----------
    prior_score_function : IMP.core.RestraintsScoringFunction over the
        restraints to treat as the prior. It must be **disjoint** from the
        likelihood scoring function -- a restraint appearing in both would be
        counted twice at lambda = 1, quietly changing the posterior. This is
        checked by name when the factory runs.
    name : label for the run log; defaults to listing the restraint names.

    Notes
    -----
    The prior is `-score`, reading IMP restraint scores as negative log
    densities exactly as the likelihood term does. There is no sampler -- an
    IMP restraint scores a state but cannot be drawn from -- so SMC
    initialization falls back to perturbing the built model, or to another
    prior's sampler when combined through `composite`.
    """

    def factory(context: PriorContext) -> Prior:
        prior_names = set(_restraint_names(prior_score_function))
        shared = prior_names.intersection(_restraint_names(context.score_function))
        if shared:
            raise ValueError(
                f"restraint_prior would double-count {sorted(shared)}: these restraints are "
                "in both the prior and the likelihood scoring function. Partition the "
                "restraints into two disjoint IMP.core.RestraintsScoringFunction objects."
            )

        jax_interface = prior_score_function._get_jax()
        radii = context.radii

        def log_prob(theta: dict) -> jnp.ndarray:
            return -jax_interface.score_func({"xyz": context.expand(theta), "r": radii})

        return Prior(
            name=name or f"restraints({','.join(sorted(prior_names))})",
            log_prob=log_prob,
            sample=None,
        )

    return factory


def composite(*factories: PriorFactory, name: Optional[str] = None) -> PriorFactory:
    """Combine priors by multiplying their densities (summing log-densities).

    The usual combination is `composite(restraint_prior(connectivity),
    bounding_box(...))`: connectivity supplies the structural knowledge, the
    box supplies both normalizability and -- being the only component that
    can be drawn from -- the SMC initial population. The sampler of the first
    component that has one is used; combining that draw with the other
    components' densities approximates a draw from the product, which SMC's
    first reweighting step then corrects for.
    """

    def factory(context: PriorContext) -> Prior:
        parts = [make(context) for make in factories]

        def log_prob(theta: dict) -> jnp.ndarray:
            total = jnp.asarray(0.0)
            for part in parts:
                total = total + part.log_prob(theta)
            return total

        sample = next((part.sample for part in parts if part.sample is not None), None)
        return Prior(
            name=name or " + ".join(part.name for part in parts),
            log_prob=log_prob,
            sample=sample,
        )

    return factory


def from_log_prob(log_prob: LogProbFn, name: str = "custom") -> PriorFactory:
    """Wrap a hand-written `theta -> scalar` log-density as a prior.

    The escape hatch for a one-off prior that none of the factories above
    cover. It has no sampler, so SMC initialization falls back to perturbing
    the built model.
    """

    def factory(context: PriorContext) -> Prior:
        del context
        return Prior(name=name, log_prob=log_prob, sample=None)

    return factory


def resolve(prior, context: PriorContext) -> Prior:
    """Normalize the wrapper's `prior=` argument into a `Prior`.

    Accepts None (meaning `flat()`), an already-built `Prior`, or a
    `PriorFactory`. A bare log-density callable is deliberately *not*
    accepted: it is indistinguishable from a factory at runtime and guessing
    wrong would silently mis-score the posterior -- wrap it in
    `from_log_prob`. The single place the API's flexibility is paid for, so
    no other module has to branch on it.
    """
    if prior is None:
        return flat()(context)
    if isinstance(prior, Prior):
        return prior
    if not callable(prior):
        raise TypeError(
            f"prior must be None, a Prior, or a PriorFactory; got {type(prior)}. "
            "To pass a bare log-density function, wrap it in priors.from_log_prob."
        )

    built = prior(context)
    if not isinstance(built, Prior):
        raise TypeError(
            "prior factory must return a priors.Prior; got "
            f"{type(built)}. A bare log-density function should be wrapped in "
            "priors.from_log_prob."
        )
    return built
