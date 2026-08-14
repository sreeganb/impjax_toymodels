"""Unit tests for smc_tempered.py (BlackJAX's `blackjax.smc.tempered`)."""

import unittest

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from impjax_toymodels import proposals, smc_particles, smc_tempered, wrapper_impjax
from toy_fixture import build_split_toy_system


class BuildRmhMcmcPairTests(unittest.TestCase):
    """The adapter that lets BlackJAX drive our SO(3)-aware proposal."""

    def test_pair_satisfies_blackjaxs_step_and_init_contract(self):
        logdensity_fn = lambda theta: -0.5 * jnp.sum(theta["x"] ** 2)
        proposal_fn = lambda key, theta: {"x": theta["x"] + 0.1 * jax.random.normal(key, theta["x"].shape)}

        step_fn, init_fn = smc_tempered.build_rmh_mcmc_pair(proposal_fn)
        self.assertIs(init_fn, blackjax.rmh.init)

        state = init_fn({"x": jnp.zeros(3)}, logdensity_fn)
        # BlackJAX calls the step function as (key, state, logdensity_fn).
        new_state, info = step_fn(jax.random.PRNGKey(0), state, logdensity_fn)
        self.assertEqual(new_state.position["x"].shape, (3,))
        self.assertIn(bool(info.is_accepted), (True, False))


class RunTemperedSmcTests(unittest.TestCase):
    def setUp(self):
        self.built, self.likelihood_sf, self.prior_sf = build_split_toy_system()
        self.context = wrapper_impjax.build_log_prob(self.built, self.likelihood_sf)
        self.proposal_fn = proposals.build_composite(
            self.context.layout,
            sigma_rotation=0.05,
            sigma_translation=2.0,
            sigma_bead=2.0,
            mode="all",
        )

    def _run(self, **kwargs):
        particles = smc_particles.initialize_particles(
            jax.random.PRNGKey(0), 12, self.context.initial_theta, None, self.proposal_fn
        )
        return smc_tempered.run_tempered_smc(
            jax.random.PRNGKey(1),
            self.context.log_prior_fn,
            self.context.log_likelihood_fn,
            self.context.log_prob_fn,
            particles,
            self.proposal_fn,
            n_mcmc_steps=3,
            verbose=False,
            **kwargs,
        )

    def test_walks_the_requested_ladder_and_tracks_one_best_per_step(self):
        n_steps = 6
        state, best_thetas, best_scores, lambdas = self._run(n_temperature_steps=n_steps)

        self.assertEqual(len(best_thetas), n_steps + 1)
        self.assertEqual(len(best_scores), n_steps + 1)
        self.assertEqual(lambdas[0], 0.0)
        self.assertEqual(lambdas[-1], 1.0)
        self.assertAlmostEqual(float(state.tempering_param), 1.0, places=6)
        self.assertEqual(smc_particles.particle_count(state.particles), 12)
        for theta in best_thetas:
            self.assertEqual(
                theta["quaternions"].shape, self.context.initial_theta["quaternions"].shape
            )

    def test_an_explicit_ladder_overrides_the_named_schedule(self):
        ladder = np.array([0.0, 0.25, 0.9, 1.0])
        _, best_thetas, _, lambdas = self._run(lambdas=ladder)
        np.testing.assert_allclose(lambdas, ladder)
        self.assertEqual(len(best_thetas), len(ladder))

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run(schedule="not-a-schedule")


if __name__ == "__main__":
    unittest.main()
