"""Unit tests for smc_particles.py, the helpers shared by all SMC variants."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impjax_toymodels import priors, proposals, smc_particles, wrapper_impjax
from toy_fixture import build_toy_system


class SmcParticlesTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.sf.evaluate(False)
        self.context = wrapper_impjax.build_log_prob(self.built, self.sf)
        self.proposal_fn = proposals.build_composite(
            self.context.layout,
            sigma_rotation=0.05,
            sigma_translation=2.0,
            sigma_bead=2.0,
            mode="all",
        )

    def _particles(self, n, seed=0):
        return smc_particles.initialize_particles(
            jax.random.PRNGKey(seed), n, self.context.initial_theta, None, self.proposal_fn
        )

    def test_particle_count_and_selection(self):
        particles = self._particles(7)
        self.assertEqual(smc_particles.particle_count(particles), 7)
        one = smc_particles.select_particle(particles, 3)
        self.assertEqual(one["quaternions"].shape, particles["quaternions"].shape[1:])

    def test_batched_scorer_matches_plain_vmap(self):
        particles = self._particles(10, seed=1)
        scorer = smc_particles.batched_scorer(self.context.log_prob_fn, batch_size=3)
        np.testing.assert_allclose(
            np.array(scorer(particles)),
            np.array(jax.vmap(self.context.log_prob_fn)(particles)),
            atol=1e-5,
        )

    def test_best_particle_selector_picks_the_argmax(self):
        particles = self._particles(9, seed=2)
        best = smc_particles.best_particle_selector(self.context.log_prob_fn, batch_size=4)
        theta, score = best(particles)
        all_scores = np.array(jax.vmap(self.context.log_prob_fn)(particles))
        self.assertAlmostEqual(score, float(all_scores.max()), places=4)
        expected = smc_particles.select_particle(particles, int(all_scores.argmax()))
        np.testing.assert_allclose(
            np.asarray(theta["bead_coords"]), np.asarray(expected["bead_coords"]), atol=1e-6
        )

    def test_perturbation_fallback_spreads_the_population(self):
        """Without a prior sampler the population is replicate-and-nudge, and
        the particles must actually differ -- identical particles would make
        the SMC weights degenerate."""
        particles = self._particles(6, seed=3)
        beads = np.asarray(particles["bead_coords"])
        self.assertEqual(beads.shape[0], 6)
        self.assertGreater(float(beads.std(axis=0).mean()), 0.0)

    def test_prior_sampler_is_used_when_available(self):
        """A prior that can be drawn from must be drawn from: SMC's lambda = 0
        distribution is the prior, so this is the statistically correct
        initialization rather than a spread of copies of one structure."""
        prior_context = priors.PriorContext(
            layout=self.context.layout,
            expand=lambda theta: theta["bead_coords"],
            radii=jnp.zeros(1),
            initial_theta=self.context.initial_theta,
            score_function=self.sf,
        )
        prior = priors.bounding_box(half_width=10.0, center=(0.0, 0.0, 0.0))(prior_context)

        particles = smc_particles.initialize_particles(
            jax.random.PRNGKey(4), 12, self.context.initial_theta, prior, self.proposal_fn
        )
        self.assertEqual(smc_particles.particle_count(particles), 12)
        # Every drawn coordinate lies in the box, which the built model's own
        # coordinates do not -- proving the draw came from the prior.
        self.assertLessEqual(float(jnp.max(jnp.abs(particles["bead_coords"]))), 10.0)

    def test_missing_prior_sampler_and_proposal_is_an_error(self):
        with self.assertRaises(ValueError):
            smc_particles.initialize_particles(
                jax.random.PRNGKey(5), 4, self.context.initial_theta, None, None
            )


if __name__ == "__main__":
    unittest.main()
