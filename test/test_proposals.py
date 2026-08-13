"""Unit tests for proposals.py, in particular the SO(3) rotation kernel.

test_rotation_step_is_exactly_reversible is the concrete check behind
Proposition 1 in doc/design.tex: negating the tangent-space perturbation
that took q_a -> q_b takes q_b exactly back to q_a, and both perturbations
have the same Gaussian density (it depends only on ||omega||). That is the
group-theoretic fact the symmetric-proposal / no-Hastings-correction claim
relies on.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impjax_toymodels import dof_layout, proposals, state_sync
from toy_fixture import build_toy_system


def _random_unit_quaternion(key):
    q = jax.random.normal(key, (4,))
    return q / jnp.linalg.norm(q)


class QuaternionMathTests(unittest.TestCase):
    def test_rotation_step_is_exactly_reversible(self):
        key = jax.random.PRNGKey(0)
        key_q, key_omega = jax.random.split(key)
        q_a = _random_unit_quaternion(key_q)[None, :]
        omega = jax.random.normal(key_omega, (1, 3)) * 0.3

        delta_q = proposals._quat_exp(omega)
        q_b = proposals._quat_multiply(delta_q, q_a)
        q_b = q_b / jnp.linalg.norm(q_b, axis=-1, keepdims=True)

        delta_q_reverse = proposals._quat_exp(-omega)
        q_a_recovered = proposals._quat_multiply(delta_q_reverse, q_b)
        q_a_recovered = q_a_recovered / jnp.linalg.norm(q_a_recovered, axis=-1, keepdims=True)

        # q and -q represent the same rotation; compare up to sign.
        same_sign = np.allclose(q_a_recovered, q_a, atol=1e-5)
        opposite_sign = np.allclose(q_a_recovered, -np.asarray(q_a), atol=1e-5)
        self.assertTrue(same_sign or opposite_sign)

    def test_rotation_step_preserves_unit_norm(self):
        key = jax.random.PRNGKey(1)
        q = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
        q_new = proposals.rotation_step(key, q, sigma=0.4)
        norms = jnp.linalg.norm(q_new, axis=-1)
        np.testing.assert_allclose(np.asarray(norms), np.ones(5), atol=1e-6)

    def test_identity_omega_gives_identity_quaternion(self):
        omega = jnp.zeros((3, 3))
        dq = proposals._quat_exp(omega)
        np.testing.assert_allclose(np.asarray(dq), np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))


class CompositeProposalTests(unittest.TestCase):
    def setUp(self):
        self.built, _ = build_toy_system()
        self.layout = dof_layout.build(self.built)
        self.theta = state_sync.extract(self.built, self.layout)

    def _proposed(self, mode):
        proposal_fn = proposals.build_composite(
            self.layout, sigma_rotation=0.1, sigma_translation=2.0, sigma_bead=1.5, mode=mode
        )
        return proposal_fn(jax.random.PRNGKey(42), self.theta)

    def test_rotation_only_mode_leaves_translations_and_beads_unchanged(self):
        new_theta = self._proposed("rotation")
        np.testing.assert_array_equal(np.asarray(new_theta["translations"]), self.theta["translations"])
        np.testing.assert_array_equal(np.asarray(new_theta["bead_coords"]), self.theta["bead_coords"])
        self.assertFalse(np.allclose(np.asarray(new_theta["quaternions"]), self.theta["quaternions"]))

    def test_beads_only_mode_leaves_rigid_body_unchanged(self):
        new_theta = self._proposed("beads")
        np.testing.assert_array_equal(np.asarray(new_theta["quaternions"]), self.theta["quaternions"])
        np.testing.assert_array_equal(np.asarray(new_theta["translations"]), self.theta["translations"])
        self.assertFalse(np.allclose(np.asarray(new_theta["bead_coords"]), self.theta["bead_coords"]))

    def test_all_mode_moves_every_group(self):
        new_theta = self._proposed("all")
        self.assertFalse(np.allclose(np.asarray(new_theta["quaternions"]), self.theta["quaternions"]))
        self.assertFalse(np.allclose(np.asarray(new_theta["translations"]), self.theta["translations"]))
        self.assertFalse(np.allclose(np.asarray(new_theta["bead_coords"]), self.theta["bead_coords"]))

    def test_proposed_state_has_matching_shapes(self):
        new_theta = self._proposed("all")
        for key in self.theta:
            self.assertEqual(np.asarray(new_theta[key]).shape, self.theta[key].shape)


if __name__ == "__main__":
    unittest.main()
