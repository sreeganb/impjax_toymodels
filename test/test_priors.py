"""Unit tests for priors.py, the user-selectable prior distributions."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impjax_toymodels import priors, wrapper_impjax
from toy_fixture import build_split_toy_system, build_toy_system


def _context_for(built, score_function) -> priors.PriorContext:
    """Build a PriorContext the same way wrapper_impjax.build_log_prob does,
    by going through build_log_prob itself and re-deriving the pieces."""
    from impjax_toymodels import dof_layout, state_sync

    layout = dof_layout.build(built)
    jax_interface = score_function._get_jax()
    template_xyz, radii = state_sync.capture_template(jax_interface)
    return priors.PriorContext(
        layout=layout,
        expand=state_sync.make_expansion_fn(layout, template_xyz),
        radii=jnp.asarray(radii),
        initial_theta=jax.tree_util.tree_map(jnp.asarray, state_sync.extract(built, layout)),
        score_function=score_function,
    )


class FlatPriorTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.sf.evaluate(False)
        self.context = _context_for(self.built, self.sf)

    def test_flat_prior_scores_zero_and_has_no_sampler(self):
        prior = priors.flat()(self.context)
        self.assertEqual(float(prior.log_prob(self.context.initial_theta)), 0.0)
        self.assertIsNone(prior.sample)

    def test_resolve_defaults_to_flat(self):
        self.assertEqual(priors.resolve(None, self.context).name, "flat")

    def test_resolve_rejects_a_bare_log_density(self):
        """A bare `theta -> scalar` is indistinguishable from a factory, so it
        must be wrapped rather than guessed at."""
        with self.assertRaises(TypeError):
            priors.resolve(jnp.asarray, self.context)
        wrapped = priors.resolve(priors.from_log_prob(lambda theta: jnp.asarray(-1.5)), self.context)
        self.assertAlmostEqual(float(wrapped.log_prob(self.context.initial_theta)), -1.5)


class BoundingBoxPriorTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.sf.evaluate(False)
        self.context = _context_for(self.built, self.sf)
        self.prior = priors.bounding_box(half_width=30.0, center=(0.0, 0.0, 0.0))(self.context)

    def test_inside_the_box_is_flat_and_outside_is_penalized(self):
        layout = self.context.layout
        inside = {
            "quaternions": jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (layout.n_rigid_bodies, 1)),
            "translations": jnp.zeros((layout.n_rigid_bodies, 3)),
            "bead_coords": jnp.zeros((layout.n_beads, 3)),
        }
        self.assertEqual(float(self.prior.log_prob(inside)), 0.0)

        outside = dict(inside, bead_coords=inside["bead_coords"].at[0, 0].set(40.0))
        # 10 units past the wall, harmonic with wall_sigma=1 -> 0.5 * 10^2.
        self.assertAlmostEqual(float(self.prior.log_prob(outside)), -50.0, places=4)

    def test_sampler_produces_in_box_coordinates_and_unit_quaternions(self):
        layout = self.context.layout
        theta = self.prior.sample(jax.random.PRNGKey(0))
        self.assertEqual(theta["quaternions"].shape, (layout.n_rigid_bodies, 4))
        self.assertEqual(theta["translations"].shape, (layout.n_rigid_bodies, 3))
        self.assertEqual(theta["bead_coords"].shape, (layout.n_beads, 3))
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(theta["quaternions"]), axis=-1), 1.0, atol=1e-6
        )
        self.assertLessEqual(float(jnp.max(jnp.abs(theta["bead_coords"]))), 30.0)
        # A draw from the prior must have prior density 1 (log 0).
        self.assertAlmostEqual(float(self.prior.log_prob(theta)), 0.0, places=5)

    def test_default_center_tracks_the_built_model(self):
        centered = priors.bounding_box(half_width=1e6)(self.context)
        theta = centered.sample(jax.random.PRNGKey(1))
        built_centroid = np.asarray(self.context.initial_theta["bead_coords"]).mean(axis=0)
        # With a huge box the draw is dominated by the box, but its centre
        # should sit near the system as built, not at the origin.
        self.assertLess(
            float(np.linalg.norm(np.asarray(theta["bead_coords"]).mean(axis=0) - built_centroid)),
            1e6,
        )


class RestraintPriorTests(unittest.TestCase):
    def setUp(self):
        self.built, self.likelihood_sf, self.prior_sf = build_split_toy_system()
        self.context = _context_for(self.built, self.likelihood_sf)

    def test_restraint_prior_reproduces_imps_own_cpu_score(self):
        """The whole point of a restraint prior: it must be the same number
        IMP would compute on the CPU, negated."""
        prior = priors.restraint_prior(self.prior_sf)(self.context)
        jax_log_prior = float(prior.log_prob(self.context.initial_theta))
        cpu_log_prior = -self.prior_sf.evaluate(False)
        self.assertAlmostEqual(jax_log_prior / cpu_log_prior, 1.0, places=5)
        self.assertIn("ConnectivityRestraint", prior.name)
        self.assertIsNone(prior.sample)

    def test_overlapping_restraints_are_rejected_rather_than_double_counted(self):
        with self.assertRaises(ValueError) as caught:
            priors.restraint_prior(self.likelihood_sf)(self.context)
        self.assertIn("double-count", str(caught.exception))

    def test_composite_sums_log_densities_and_borrows_a_sampler(self):
        restraint = priors.restraint_prior(self.prior_sf)
        box = priors.bounding_box(half_width=25.0, center=(0.0, 0.0, 0.0))
        combined = priors.composite(restraint, box)(self.context)

        theta = self.context.initial_theta
        expected = float(restraint(self.context).log_prob(theta)) + float(
            box(self.context).log_prob(theta)
        )
        # Relative tolerance: these are float32 sums of order 1e5, so an
        # absolute comparison would be testing float32's mantissa, not the code.
        np.testing.assert_allclose(float(combined.log_prob(theta)), expected, rtol=1e-5)
        # restraint_prior has no sampler; the box's is picked up instead.
        self.assertIsNotNone(combined.sample)

    def test_wrapper_splits_prior_and_likelihood_without_double_counting(self):
        context = wrapper_impjax.build_log_prob(
            self.built, self.likelihood_sf, prior=priors.restraint_prior(self.prior_sf)
        )
        theta = context.initial_theta
        np.testing.assert_allclose(
            float(context.log_prob_fn(theta)),
            float(context.log_prior_fn(theta)) + float(context.log_likelihood_fn(theta)),
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
