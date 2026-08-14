"""Unit tests for smc_adaptive_tempered.py (`blackjax.smc.adaptive_tempered`).

The behaviour that distinguishes this sampler from the fixed-ladder ones is
that it *chooses* its temperature ladder, so the tests here are about the
ladder itself: that it is monotone, that it reaches 1, that its step sizes
respond to how sharp the likelihood is, and that the run length is an output
rather than an input.

A synthetic Gaussian likelihood over the reduced-state pytree is used for the
ladder tests rather than an IMP system: it makes "how sharp is the
likelihood" a knob the test controls directly. The IMP path is covered
end-to-end in test_run_smc_sampling.py.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from impjax_toymodels import smc_adaptive_tempered


def _synthetic_problem(sharpness: float):
    """A Gaussian likelihood centred away from a standard-normal prior.

    Returns (log_prior_fn, log_likelihood_fn, log_prob_fn, proposal_fn) over a
    one-leaf pytree standing in for a reduced state.
    """
    log_prior_fn = lambda theta: -0.5 * jnp.sum(theta["x"] ** 2)
    log_likelihood_fn = lambda theta: -0.5 * jnp.sum(((theta["x"] - 3.0) / sharpness) ** 2)
    log_prob_fn = lambda theta: log_prior_fn(theta) + log_likelihood_fn(theta)

    def proposal_fn(key, theta):
        return {"x": theta["x"] + 0.4 * jax.random.normal(key, theta["x"].shape)}

    return log_prior_fn, log_likelihood_fn, log_prob_fn, proposal_fn


def _run(sharpness=0.5, n_particles=200, seed=0, **kwargs):
    log_prior_fn, log_likelihood_fn, log_prob_fn, proposal_fn = _synthetic_problem(sharpness)
    particles = {"x": jax.random.normal(jax.random.PRNGKey(seed), (n_particles, 2))}
    return smc_adaptive_tempered.run_adaptive_tempered_smc(
        jax.random.PRNGKey(seed + 1),
        log_prior_fn,
        log_likelihood_fn,
        log_prob_fn,
        particles,
        proposal_fn,
        n_mcmc_steps=5,
        verbose=False,
        **kwargs,
    )


class AdaptiveLadderTests(unittest.TestCase):
    def test_ladder_is_monotone_and_terminates_at_one(self):
        state, best_thetas, best_scores, lambdas = _run(target_ess=0.6)

        self.assertEqual(lambdas[0], 0.0)
        self.assertAlmostEqual(float(lambdas[-1]), 1.0, places=6)
        self.assertAlmostEqual(float(state.tempering_param), 1.0, places=6)
        self.assertTrue(np.all(np.diff(lambdas) > 0), f"ladder not increasing: {lambdas}")
        # Run length is an output: one recorded state per step, plus the initial one.
        self.assertEqual(len(best_thetas), len(lambdas))
        self.assertEqual(len(best_scores), len(lambdas))

    def test_ladder_is_not_uniform(self):
        """The point of adapting: increments vary with how fast the tempered
        target is changing, unlike a linear schedule's constant stride."""
        _, _, _, lambdas = _run(target_ess=0.7)
        increments = np.diff(lambdas)
        self.assertGreater(len(increments), 2)
        self.assertGreater(float(increments.std()), 1e-3)

    def test_a_sharper_likelihood_needs_more_temperature_steps(self):
        _, _, _, easy = _run(sharpness=2.0, target_ess=0.7, seed=10)
        _, _, _, hard = _run(sharpness=0.2, target_ess=0.7, seed=10)
        self.assertGreater(len(hard), len(easy), f"easy={len(easy)} hard={len(hard)}")

    def test_particles_move_towards_the_likelihood_mode(self):
        state, _, _, _ = _run(sharpness=0.5, target_ess=0.6)
        self.assertGreater(float(jnp.mean(state.particles["x"])), 2.0)

    def test_max_steps_stops_the_run_short_of_one(self):
        state, _, _, lambdas = _run(sharpness=0.2, target_ess=0.95, max_steps=2)
        self.assertEqual(len(lambdas) - 1, 2)
        self.assertLess(float(state.tempering_param), 1.0)

    def test_target_ess_must_be_a_fraction(self):
        for bad in (0.0, 1.0, 1.5, -0.2):
            with self.assertRaises(ValueError):
                _run(target_ess=bad)


if __name__ == "__main__":
    unittest.main()
