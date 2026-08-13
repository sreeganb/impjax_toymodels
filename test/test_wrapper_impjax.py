"""End-to-end tests for wrapper_impjax.py against a real IMP toy system.

These are integration tests: they exercise the whole pipeline (IMP model ->
dof_layout -> state_sync -> proposals -> BlackJAX RMH -> sync back), not a
single module in isolation. Module-level correctness is covered by
test_dof_layout.py, test_state_sync.py, and test_proposals.py.
"""

import csv
import os
import tempfile
import unittest

import jax
import numpy as np
import RMF

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

    def test_rmf_path_produces_full_pipeline_output(self):
        """The non-negotiable end-to-end case: one call builds, samples, writes
        an RMF3 trajectory + stat file, and logs a run summary including the
        acceptance rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rmf_path = os.path.join(tmpdir, "run.rmf3")
            log_path = os.path.join(tmpdir, "run.log")

            positions, log_probs, acceptance_rate = wrapper_impjax.run_sampling(
                self.built,
                self.sf,
                jax.random.PRNGKey(5),
                n_steps=12,
                mode="all",
                sigma_rotation=0.02,
                sigma_translation=0.5,
                sigma_bead=0.5,
                rmf_path=rmf_path,
                log_path=log_path,
                verbose=False,
            )

            handle = RMF.open_rmf_file_read_only(rmf_path)
            self.assertEqual(handle.get_number_of_frames(), len(positions))

            stat_path = os.path.join(tmpdir, "run_stats.csv")
            self.assertTrue(os.path.exists(stat_path))
            with open(stat_path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows) - 1, len(positions))

            self.assertTrue(os.path.exists(log_path))
            with open(log_path) as f:
                log_contents = f.read()
            self.assertIn("run_sampling starting", log_contents)
            self.assertIn("acceptance_rate=", log_contents)
            self.assertIn(f"{100 * acceptance_rate:.1f}%", log_contents)

            # rmf_path implies the model ends up holding the last written state.
            layout = dof_layout.build(self.built)
            theta_now = state_sync.extract(self.built, layout)
            np.testing.assert_allclose(
                theta_now["bead_coords"], positions[-1]["bead_coords"], atol=1e-6
            )

    def test_explicit_stat_path_is_honored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rmf_path = os.path.join(tmpdir, "traj.rmf3")
            stat_path = os.path.join(tmpdir, "custom_stats.csv")
            wrapper_impjax.run_sampling(
                self.built,
                self.sf,
                jax.random.PRNGKey(6),
                n_steps=5,
                mode="beads",
                sigma_bead=1.0,
                rmf_path=rmf_path,
                stat_path=stat_path,
                verbose=False,
            )
            self.assertTrue(os.path.exists(stat_path))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "traj_stats.csv")))


if __name__ == "__main__":
    unittest.main()
