"""End-to-end tests for wrapper_impjax.py against a real IMP toy system.

These are integration tests: they exercise the whole pipeline (IMP model ->
dof_layout -> state_sync -> proposals -> BlackJAX RMH -> sync back), not a
single module in isolation. Module-level correctness is covered by
test_dof_layout.py, test_state_sync.py, and test_proposals.py.
"""

import unittest

import jax
import numpy as np

from impjax_toymodels import dof_layout, state_sync, wrapper_impjax
from toy_fixture import build_toy_system


class WrapperImpjaxTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.sf.evaluate(False)  # materialize IMP's JAX export

    def test_build_log_prob_matches_imp_score_at_initial_state(self):
        context = wrapper_impjax.build_log_prob(self.built, self.sf)
        log_prob = float(context.log_prob_fn(context.initial_theta))
        self.assertAlmostEqual(log_prob, -self.sf.evaluate(False), places=4)

    def test_run_sampling_all_mode_produces_requested_samples(self):
        positions, log_probs, acceptance_rate = wrapper_impjax.run_sampling(
            self.built,
            self.sf,
            jax.random.PRNGKey(0),
            n_steps=40,
            mode="all",
            sigma_rotation=0.02,
            sigma_translation=0.5,
            sigma_bead=0.5,
            verbose=False,
        )
        self.assertEqual(len(positions), 40)
        self.assertEqual(log_probs.shape, (40,))
        self.assertGreaterEqual(acceptance_rate, 0.0)
        self.assertLessEqual(acceptance_rate, 1.0)

    def test_beads_only_mode_never_moves_rigid_body(self):
        layout = dof_layout.build(self.built)
        theta0 = state_sync.extract(self.built, layout)

        positions, _, _ = wrapper_impjax.run_sampling(
            self.built,
            self.sf,
            jax.random.PRNGKey(2),
            n_steps=20,
            mode="beads",
            sigma_bead=1.0,
            sync_back=False,
            verbose=False,
        )
        for theta in positions:
            np.testing.assert_allclose(theta["quaternions"], theta0["quaternions"], atol=1e-6)
            np.testing.assert_allclose(theta["translations"], theta0["translations"], atol=1e-6)

    def test_sync_back_writes_final_sample_into_imp_model(self):
        positions, _, _ = wrapper_impjax.run_sampling(
            self.built,
            self.sf,
            jax.random.PRNGKey(3),
            n_steps=15,
            mode="beads",
            sigma_bead=2.0,
            sync_back=True,
            verbose=False,
        )
        layout = dof_layout.build(self.built)
        theta_now = state_sync.extract(self.built, layout)
        np.testing.assert_allclose(theta_now["bead_coords"], positions[-1]["bead_coords"], atol=1e-6)

    def test_sync_back_false_leaves_imp_model_untouched(self):
        layout = dof_layout.build(self.built)
        theta0 = state_sync.extract(self.built, layout)
        wrapper_impjax.run_sampling(
            self.built,
            self.sf,
            jax.random.PRNGKey(4),
            n_steps=15,
            mode="beads",
            sigma_bead=5.0,
            sync_back=False,
            verbose=False,
        )
        theta_after = state_sync.extract(self.built, layout)
        np.testing.assert_allclose(theta_after["bead_coords"], theta0["bead_coords"])


if __name__ == "__main__":
    unittest.main()
