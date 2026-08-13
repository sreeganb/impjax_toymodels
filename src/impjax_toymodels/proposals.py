"""SO(3)-compliant, detailed-balance-preserving proposal kernels.

Three independent kernels, one per degree-of-freedom group (doc/design.tex
Section 5), composed into a single proposal function that BlackJAX's plain
Metropolis kernel (`blackjax.rmh(logdensity_fn, proposal)`) can drive with
NO Hastings correction, because every kernel here is symmetric by
construction:

  - rotation_step:    Gaussian step in the SO(3) tangent space (the Lie
                       algebra), mapped back via the quaternion exponential
                       and composed on the left; symmetric because the
                       tangent-space Gaussian is symmetric under negation and
                       left-multiplication preserves the Haar measure on
                       SO(3) (see doc/design.tex Proposition 1).
  - translation_step: ordinary additive isotropic Gaussian; symmetric
                       trivially.
  - build_composite:  masks out whichever of {rotation, rigid translation,
                       bead translation} the chosen sampling mode excludes,
                       replacing them with an identity step (also trivially
                       symmetric), matching the five user-selectable modes
                       required by planning.md.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from .dof_layout import ModeMask, SystemLayout, mode_mask

ProposalFn = Callable[[jax.Array, dict], dict]


def _quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product q1 * q2, both (..., 4) with layout (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.stack([w, x, y, z], axis=-1)


def _quat_exp(omega: jnp.ndarray) -> jnp.ndarray:
    """Quaternion exponential map exp: so(3) = R^3 -> unit quaternions.

    omega is a tangent vector at the identity; ||omega|| is the rotation
    angle and omega/||omega|| the axis. The angle=0 case (identity rotation)
    is a removable singularity, handled with a safe division.
    """
    angle = jnp.linalg.norm(omega, axis=-1, keepdims=True)
    safe_angle = jnp.where(angle > 0, angle, 1.0)
    axis = jnp.where(angle > 0, omega / safe_angle, jnp.zeros_like(omega))
    half = angle / 2.0
    w = jnp.cos(half)
    xyz = jnp.sin(half) * axis
    return jnp.concatenate([w, xyz], axis=-1)


def rotation_step(key: jax.Array, quaternions: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """SO(3)-compliant proposal for a batch of K rigid-body quaternions.

    Draws omega ~ N(0, sigma^2 I_3) per rigid body, composes
    q' = exp(omega) (x) q, and renormalizes to absorb floating-point drift.
    """
    omega = jax.random.normal(key, quaternions.shape[:-1] + (3,)) * sigma
    delta_q = _quat_exp(omega)
    proposed = _quat_multiply(delta_q, quaternions)
    return proposed / jnp.linalg.norm(proposed, axis=-1, keepdims=True)


def translation_step(key: jax.Array, positions: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Symmetric additive Gaussian step, shared by rigid-body translations
    and flexible-bead moves (different sigma per doc/design.tex Section 5.1,
    reflecting their different physical scales)."""
    return positions + sigma * jax.random.normal(key, positions.shape)


def build_composite(
    layout: SystemLayout,
    sigma_rotation: float,
    sigma_translation: float,
    sigma_bead: float,
    mode: str = "all",
) -> ProposalFn:
    """Assemble the single proposal function the wrapper hands to BlackJAX.

    `mode` selects which of the five planning.md sampling modes is active
    (see dof_layout.mode_mask); disabled groups are left unchanged, which is
    itself a symmetric (identity) sub-proposal.
    """
    mask: ModeMask = mode_mask(mode)
    has_rigid = layout.n_rigid_bodies > 0
    has_beads = layout.n_beads > 0

    def proposal(key: jax.Array, theta: dict) -> dict:
        key_rot, key_trans, key_bead = jax.random.split(key, 3)

        new_q = theta["quaternions"]
        if mask.rotate_rigid and has_rigid:
            new_q = rotation_step(key_rot, theta["quaternions"], sigma_rotation)

        new_t = theta["translations"]
        if mask.translate_rigid and has_rigid:
            new_t = translation_step(key_trans, theta["translations"], sigma_translation)

        new_x = theta["bead_coords"]
        if mask.translate_beads and has_beads:
            new_x = translation_step(key_bead, theta["bead_coords"], sigma_bead)

        return {"quaternions": new_q, "translations": new_t, "bead_coords": new_x}

    return proposal
