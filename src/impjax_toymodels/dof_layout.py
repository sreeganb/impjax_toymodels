"""Static index bookkeeping between IMP particles and the reduced sampling state.

Builds a `SystemLayout` once per `BuiltSystem`, fixing (for the lifetime of a
sampling run) which particles are sampled, in what order, and with what
per-rigid-body local (frame-relative) coordinates. Every other module in the
IMP<->JAX bridge (`state_sync.py`, `proposals.py`) is keyed off this ordering,
so it must be built exactly once and reused, never re-derived mid-run.

The reduced state this layout describes is the pytree

    theta = {"quaternions": (K, 4), "translations": (K, 3), "bead_coords": (M, 3)}

for K rigid bodies and M flexible beads (see doc/design.tex, Section 2).
BlackJAX's samplers accept this pytree directly as a position, so
`flatten`/`unflatten` below exist only for compact storage (the GPU IO
buffer, stat files) -- they are not required to drive the sampler itself.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import IMP.atom
import IMP.core

SAMPLING_MODES = ("rotation", "translation", "rigid", "beads", "all")


@dataclass(frozen=True)
class RigidBodyLayout:
    """Index bookkeeping for a single rigid body's members."""

    rigid_body: IMP.core.RigidBody
    member_particle_indexes: np.ndarray  # (n_members,) int64, row into the full xyz array
    local_coordinates: np.ndarray  # (n_members, 3) float64, frame-relative, fixed for the run


@dataclass(frozen=True)
class SystemLayout:
    """Ordered index map between a BuiltSystem and the reduced sampling state."""

    rigid_bodies: Tuple[RigidBodyLayout, ...]
    bead_particles: Tuple[IMP.atom.Hierarchy, ...]  # ordered flexible-bead particle handles
    bead_particle_indexes: np.ndarray  # (n_beads,) int64, row into the full xyz array

    @property
    def n_rigid_bodies(self) -> int:
        return len(self.rigid_bodies)

    @property
    def n_beads(self) -> int:
        return int(self.bead_particle_indexes.shape[0])

    @property
    def flat_size(self) -> int:
        """Dimension of the flattened state: 4 (quaternion) + 3 (translation) per
        rigid body, plus 3 (coordinate) per bead."""
        return 7 * self.n_rigid_bodies + 3 * self.n_beads


@dataclass(frozen=True)
class ModeMask:
    """Which degree-of-freedom groups a sampling mode moves.

    Corresponds to the five user-selectable modes required by planning.md:
    rotation-only, translation-only, rigid (rotation+translation), beads-only,
    and all. Disabled groups are proposed with a zero step, i.e. an identity
    move, which trivially preserves detailed balance (doc/design.tex Section
    5.3).
    """

    rotate_rigid: bool
    translate_rigid: bool
    translate_beads: bool


def mode_mask(mode: str) -> ModeMask:
    """Resolve a sampling-mode name to the DOF groups it moves."""
    if mode == "rotation":
        return ModeMask(rotate_rigid=True, translate_rigid=False, translate_beads=False)
    if mode == "translation":
        return ModeMask(rotate_rigid=False, translate_rigid=True, translate_beads=False)
    if mode == "rigid":
        return ModeMask(rotate_rigid=True, translate_rigid=True, translate_beads=False)
    if mode == "beads":
        return ModeMask(rotate_rigid=False, translate_rigid=False, translate_beads=True)
    if mode == "all":
        return ModeMask(rotate_rigid=True, translate_rigid=True, translate_beads=True)
    raise ValueError(f"Unknown sampling mode {mode!r}; choose from {SAMPLING_MODES}")


def build(built_system) -> SystemLayout:
    """Construct a SystemLayout from a BuiltSystem, fixing sampling order for the run.

    Rigid-body member particle indexes and local (frame-relative) coordinates
    are captured here once; they are treated as run constants everywhere
    downstream (they do not change as the rigid body's pose is resampled).
    """
    rigid_bodies, beads = built_system.rigid_bodies_and_beads()

    rb_layouts = []
    for rb in rigid_bodies:
        members = rb.get_rigid_members()
        member_idx = np.array(
            [int(member.get_particle_index()) for member in members], dtype=np.int64
        )
        local = np.array(
            [tuple(member.get_internal_coordinates()) for member in members],
            dtype=np.float64,
        )
        rb_layouts.append(RigidBodyLayout(rb, member_idx, local))

    bead_idx = np.array([int(bead.get_particle_index()) for bead in beads], dtype=np.int64)
    return SystemLayout(tuple(rb_layouts), tuple(beads), bead_idx)


def flatten(theta, layout: SystemLayout) -> np.ndarray:
    """Pack a reduced-state pytree into a single flat vector (for storage/IO only)."""
    quats = np.asarray(theta["quaternions"]).reshape(layout.n_rigid_bodies, 4)
    trans = np.asarray(theta["translations"]).reshape(layout.n_rigid_bodies, 3)
    beads = np.asarray(theta["bead_coords"]).reshape(layout.n_beads, 3)
    return np.concatenate([quats.reshape(-1), trans.reshape(-1), beads.reshape(-1)])


def unflatten(flat: np.ndarray, layout: SystemLayout) -> dict:
    """Inverse of `flatten`: rebuild the reduced-state pytree from a flat vector."""
    flat = np.asarray(flat)
    n_rb = layout.n_rigid_bodies
    n_beads = layout.n_beads
    q_end = 4 * n_rb
    t_end = q_end + 3 * n_rb
    return {
        "quaternions": flat[:q_end].reshape(n_rb, 4),
        "translations": flat[q_end:t_end].reshape(n_rb, 3),
        "bead_coords": flat[t_end:].reshape(n_beads, 3),
    }
