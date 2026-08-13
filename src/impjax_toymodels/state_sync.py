"""Bridge between a live IMP model and the reduced JAX sampling state.

Three operations, matching doc/design.tex Sections 3-4:

  extract(built, layout)          IMP model  -> reduced state theta (numpy)
  capture_template(ji)            IMP model  -> frozen (xyz, r) baseline, once
  make_expansion_fn(layout, ...)  reduced state theta -> full per-particle xyz
                                   (the map Phi; JIT-traceable, used inside the
                                   wrapper's log-density function)
  apply(theta, layout, built)     reduced state theta -> write back into IMP

`ji.get_jax_model()["xyz"]` is a live view aliased to IMP's internal particle
storage (verified empirically -- see doc/design.tex, Assumption 1): mutating
the model changes previously-fetched dicts too. `capture_template` is the
only place that array is touched; it is copied immediately so everything
downstream is ordinary immutable JAX.
"""

from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np

import IMP
import IMP.algebra
import IMP.core

from .dof_layout import SystemLayout


def extract(built_system, layout: SystemLayout) -> dict:
    """Read the current reduced state (quaternions, translations, bead coords)
    out of the live IMP model."""
    quats = np.empty((layout.n_rigid_bodies, 4), dtype=np.float64)
    trans = np.empty((layout.n_rigid_bodies, 3), dtype=np.float64)
    for k, rb_layout in enumerate(layout.rigid_bodies):
        transform = rb_layout.rigid_body.get_reference_frame().get_transformation_to()
        quats[k] = transform.get_rotation().get_quaternion()
        trans[k] = transform.get_translation()

    beads = np.empty((layout.n_beads, 3), dtype=np.float64)
    for m, particle in enumerate(layout.bead_particles):
        beads[m] = IMP.core.XYZ(particle).get_coordinates()

    return {"quaternions": quats, "translations": trans, "bead_coords": beads}


def capture_template(jax_interface) -> Tuple[np.ndarray, np.ndarray]:
    """Snapshot `ji.get_jax_model()` into ordinary (non-aliased) numpy arrays.

    Returns (template_xyz, r): the full per-particle coordinate array at
    build time (used to fill in rows that are not part of the sampled
    state -- see doc/design.tex Assumption 3) and the static per-particle
    radii `score_func` expects alongside "xyz".
    """
    model_dict = jax_interface.get_jax_model()
    return np.array(model_dict["xyz"]), np.array(model_dict["r"])


def make_expansion_fn(layout: SystemLayout, template_xyz: np.ndarray) -> Callable[[dict], jnp.ndarray]:
    """Build Phi: reduced state theta -> full (N, 3) per-particle xyz array.

    The returned function closes over the layout's (fixed) member indexes
    and local coordinates and the frozen template, so it takes only `theta`
    and is safe to call under `jax.jit`/`jax.grad`.
    """
    base = jnp.asarray(template_xyz)
    member_indexes = [jnp.asarray(rb.member_particle_indexes) for rb in layout.rigid_bodies]
    local_coords = [jnp.asarray(rb.local_coordinates) for rb in layout.rigid_bodies]
    bead_indexes = jnp.asarray(layout.bead_particle_indexes)

    def _rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Active rotation of v (n, 3) by unit quaternion q = (w, x, y, z)."""
        w, qv = q[0], q[1:4]
        uv = jnp.cross(qv, v)
        uuv = jnp.cross(qv, uv)
        return v + 2.0 * (w * uv + uuv)

    def expand(theta: dict) -> jnp.ndarray:
        xyz = base
        for k in range(layout.n_rigid_bodies):
            q_k = theta["quaternions"][k]
            t_k = theta["translations"][k]
            global_k = _rotate(q_k, local_coords[k]) + t_k
            xyz = xyz.at[member_indexes[k]].set(global_k)
        xyz = xyz.at[bead_indexes].set(theta["bead_coords"])
        return xyz

    return expand


def apply(theta: dict, layout: SystemLayout, built_system) -> None:
    """Write a reduced state back into the live IMP model.

    Rigid bodies are moved via `set_reference_frame` (rotation + translation
    of the whole body, which also moves every member); beads are moved
    directly. Quaternions are renormalized defensively before being handed
    to IMP, since repeated composition (see proposals.py) can accumulate
    floating-point drift away from unit norm.
    """
    quats = np.asarray(theta["quaternions"])
    trans = np.asarray(theta["translations"])
    for k, rb_layout in enumerate(layout.rigid_bodies):
        q = quats[k]
        q = q / np.linalg.norm(q)
        rotation = IMP.algebra.Rotation3D(*q.tolist())
        transformation = IMP.algebra.Transformation3D(rotation, IMP.algebra.Vector3D(*trans[k].tolist()))
        rb_layout.rigid_body.set_reference_frame(IMP.algebra.ReferenceFrame3D(transformation))

    beads = np.asarray(theta["bead_coords"])
    for m, particle in enumerate(layout.bead_particles):
        IMP.core.XYZ(particle).set_coordinates(IMP.algebra.Vector3D(*beads[m].tolist()))
