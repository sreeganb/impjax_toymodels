"""Unit tests for smc_fixed_schedule.py, the untuned base SMC over a pytree state."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impjax_toymodels import proposals, smc_fixed_schedule, smc_particles, wrapper_impjax
from toy_fixture import build_toy_system


class SmcFixedScheduleTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.sf.evaluate(False)
        self.context = wrapper_impjax.build_log_prob(self.built, self.sf)
        self.proposal_fn = proposals.build_composite(
            self.context.layout, sigma_rotation=0.05, sigma_translation=2.0, sigma_bead=2.0, mode="all"
        )

    def _initial_particles(self, n_particles, key):
        replicated = jax.tree_util.tree_map(
            lambda leaf: jnp.tile(leaf[None, ...], (n_particles,) + (1,) * leaf.ndim),
            self.context.initial_theta,
        )
        keys = jax.random.split(key, n_particles)
        return jax.vmap(self.proposal_fn)(keys, replicated)

    def test_run_fixed_schedule_smc_returns_one_best_theta_per_step(self):
        n_particles = 12
        n_temperature_steps = 6
        key = jax.random.PRNGKey(2)
        spread_key, run_key = jax.random.split(key)
        initial_particles = self._initial_particles(n_particles, spread_key)

        log_prior_fn = lambda theta: jnp.asarray(0.0)

        state, best_thetas, best_scores, lambdas = smc_fixed_schedule.run_fixed_schedule_smc(
            run_key,
            log_prior_fn,
            self.context.log_likelihood_fn,
            self.context.log_prob_fn,
            initial_particles,
            self.proposal_fn,
            n_temperature_steps=n_temperature_steps,
            n_mcmc_steps=3,
            verbose=False,
        )

        self.assertEqual(len(best_thetas), n_temperature_steps + 1)
        self.assertEqual(len(best_scores), n_temperature_steps + 1)
        self.assertEqual(lambdas[0], 0.0)
        self.assertEqual(lambdas[-1], 1.0)
        for theta in best_thetas:
            self.assertEqual(theta["quaternions"].shape, self.context.initial_theta["quaternions"].shape)
        # Final SMC particle population still has n_particles members.
        self.assertEqual(smc_particles.particle_count(state.particles), n_particles)


if __name__ == "__main__":
    unittest.main()
