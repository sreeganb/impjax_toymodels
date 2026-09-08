"""The IMP<->JAX bridge on a system with rigid bodies but no flexible beads.

Every bridge module carries a flexible-bead branch: `bead_coords` in the
reduced state, a bead sub-proposal, a bead row-set inside the expansion map
Phi, a bead term in the box prior.  A rigid-body docking system exercises all
of them at size zero -- empty index arrays, (0, 3) coordinate arrays, empty
`.at[].set()` updates -- and a shape or dtype slip on that path is invisible to
every other test in the suite, because every other fixture has beads.

So these tests are not about docking specifically; they hold the M = 0 path
open for any system built entirely out of rigid bodies.
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from toy_fixture import build_rigid_only_system  # noqa: E402

from impjax_toymodels import (  # noqa: E402
    dof_layout,
    priors,
    proposals,
    smc_particles,
    state_sync,
)


class ZeroBeadLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built, cls.score_function = build_rigid_only_system()
        cls.layout = dof_layout.build(cls.built)
        cls.theta = state_sync.extract(cls.built, cls.layout)
        cls.template, cls.radii = state_sync.capture_template(
            cls.score_function._get_jax())
        # staticmethod, or attribute access would bind Phi as a method and
        # pass `self` in as its theta.
        cls.expand = staticmethod(state_sync.make_expansion_fn(cls.layout, cls.template))

    def test_layout_reports_two_rigid_bodies_and_no_beads(self):
        self.assertEqual(self.layout.n_rigid_bodies, 2)
        self.assertEqual(self.layout.n_beads, 0)
        # 4 (quaternion) + 3 (translation) per body, nothing for beads.
        self.assertEqual(self.layout.flat_size, 14)

    def test_bead_index_array_is_empty_but_still_integer_typed(self):
        # An empty float array here would silently break the `.at[idx].set()`
        # in the expansion map, which requires integer indexes.
        indexes = self.layout.bead_particle_indexes
        self.assertEqual(indexes.shape, (0,))
        self.assertTrue(np.issubdtype(indexes.dtype, np.integer))

    def test_extracted_state_has_an_empty_bead_block(self):
        self.assertEqual(np.asarray(self.theta["quaternions"]).shape, (2, 4))
        self.assertEqual(np.asarray(self.theta["translations"]).shape, (2, 3))
        self.assertEqual(np.asarray(self.theta["bead_coords"]).shape, (0, 3))

    def test_flatten_unflatten_round_trips(self):
        flat = dof_layout.flatten(self.theta, self.layout)
        self.assertEqual(flat.shape, (14,))
        restored = dof_layout.unflatten(flat, self.layout)
        for key in ("quaternions", "translations", "bead_coords"):
            np.testing.assert_allclose(
                np.asarray(restored[key]), np.asarray(self.theta[key]), atol=1e-12)

    def test_expansion_reproduces_imp_coordinates(self):
        """Phi(theta) must agree with what IMP itself holds for every member.

        The real check on the zero-bead path: the empty bead update must leave
        the array untouched rather than corrupting it.
        """
        expanded = np.asarray(self.expand(self.theta))
        import IMP.core
        for rigid_body in self.layout.rigid_bodies:
            for index, member in zip(rigid_body.member_particle_indexes,
                                     rigid_body.rigid_body.get_rigid_members()):
                actual = np.asarray(list(IMP.core.XYZ(member).get_coordinates()))
                np.testing.assert_allclose(expanded[index], actual, atol=1e-6)

    def test_score_function_evaluates_on_an_expanded_state(self):
        score_func = self.score_function._get_jax().score_func
        score = score_func({"xyz": self.expand(self.theta),
                            "r": jnp.asarray(self.radii)})
        self.assertTrue(np.isfinite(float(score)))

    def test_proposal_moves_rigid_bodies_and_leaves_the_bead_block_empty(self):
        proposal = proposals.build_composite(
            self.layout, sigma_rotation=0.05, sigma_translation=1.0,
            sigma_bead=1.0, mode="all")
        theta = {k: jnp.asarray(v) for k, v in self.theta.items()}
        proposed = proposal(jax.random.PRNGKey(0), theta)

        self.assertEqual(np.asarray(proposed["bead_coords"]).shape, (0, 3))
        self.assertFalse(np.allclose(np.asarray(proposed["quaternions"]),
                                     np.asarray(theta["quaternions"])))
        self.assertFalse(np.allclose(np.asarray(proposed["translations"]),
                                     np.asarray(theta["translations"])))
        # Quaternions must stay on the unit sphere -- the SO(3) guarantee.
        norms = np.linalg.norm(np.asarray(proposed["quaternions"]), axis=-1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)

    def test_beads_mode_is_an_identity_move_when_there_are_no_beads(self):
        """`--mode beads` on a bead-free system must be a no-op, not an error."""
        proposal = proposals.build_composite(
            self.layout, 0.05, 1.0, 1.0, mode="beads")
        theta = {k: jnp.asarray(v) for k, v in self.theta.items()}
        proposed = proposal(jax.random.PRNGKey(1), theta)
        for key in ("quaternions", "translations", "bead_coords"):
            np.testing.assert_allclose(np.asarray(proposed[key]),
                                       np.asarray(theta[key]), atol=1e-12)

    def test_apply_round_trips_through_the_live_imp_model(self):
        moved = dict(self.theta)
        moved["translations"] = np.asarray(self.theta["translations"]) + 5.0
        state_sync.apply(moved, self.layout, self.built)
        read_back = state_sync.extract(self.built, self.layout)
        np.testing.assert_allclose(
            np.asarray(read_back["translations"]),
            np.asarray(moved["translations"]), atol=1e-6)
        # Restore, so ordering between tests cannot matter.
        state_sync.apply(self.theta, self.layout, self.built)

    def test_bounding_box_prior_handles_an_empty_bead_block(self):
        context = priors.PriorContext(
            layout=self.layout, expand=self.expand, radii=jnp.asarray(self.radii),
            initial_theta=self.theta, score_function=self.score_function)
        prior = priors.bounding_box(half_width=100.0)(context)

        theta = {k: jnp.asarray(v) for k, v in self.theta.items()}
        self.assertTrue(np.isfinite(float(prior.log_prob(theta))))

        drawn = prior.sample(jax.random.PRNGKey(2))
        self.assertEqual(np.asarray(drawn["bead_coords"]).shape, (0, 3))
        self.assertEqual(np.asarray(drawn["quaternions"]).shape, (2, 4))

    def test_smc_particle_initialization_from_a_samplable_prior(self):
        context = priors.PriorContext(
            layout=self.layout, expand=self.expand, radii=jnp.asarray(self.radii),
            initial_theta=self.theta, score_function=self.score_function)
        prior = priors.bounding_box(half_width=100.0)(context)

        particles = smc_particles.initialize_particles(
            jax.random.PRNGKey(3), n_particles=8, initial_theta=self.theta, prior=prior)
        self.assertEqual(smc_particles.particle_count(particles), 8)
        self.assertEqual(np.asarray(particles["bead_coords"]).shape, (8, 0, 3))
        self.assertEqual(np.asarray(particles["translations"]).shape, (8, 2, 3))

    def test_smc_particle_initialization_via_the_proposal_fallback(self):
        """The no-samplable-prior path also has to survive an empty bead block."""
        proposal = proposals.build_composite(self.layout, 0.05, 1.0, 1.0, mode="all")
        particles = smc_particles.initialize_particles(
            jax.random.PRNGKey(4), n_particles=5, initial_theta=self.theta,
            prior=None, proposal_fn=proposal)
        self.assertEqual(smc_particles.particle_count(particles), 5)
        self.assertEqual(np.asarray(particles["bead_coords"]).shape, (5, 0, 3))


if __name__ == "__main__":
    unittest.main()
