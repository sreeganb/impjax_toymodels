"""End-to-end tests for wrapper_impjax.run_smc_sampling."""

import csv
import os
import tempfile
import unittest

import jax
import numpy as np
import RMF

from impjax_toymodels import dof_layout, state_sync, wrapper_impjax
from toy_fixture import build_toy_system


class RunSmcSamplingTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.sf.evaluate(False)

    def test_best_scores_are_non_decreasing_across_the_anneal(self):
        """The best particle's log-posterior should not get worse as lambda -> 1
        on this easy toy system (monotone improvement is the whole point of
        tracking the best-of-population at each step)."""
        best_thetas, best_scores, lambdas = wrapper_impjax.run_smc_sampling(
            self.built,
            self.sf,
            jax.random.PRNGKey(0),
            n_particles=15,
            n_temperature_steps=8,
            n_mcmc_steps=4,
            sigma_rotation=0.05,
            sigma_translation=1.0,
            sigma_bead=1.0,
            verbose=False,
        )
        self.assertEqual(len(best_thetas), 9)
        self.assertEqual(len(best_scores), 9)
        self.assertEqual(lambdas[0], 0.0)
        self.assertEqual(lambdas[-1], 1.0)
        diffs = np.diff(best_scores)
        self.assertTrue(np.all(diffs >= -1e-6), f"best score decreased somewhere: {best_scores}")

    def test_rmf_path_writes_one_frame_per_temperature_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rmf_path = os.path.join(tmpdir, "smc.rmf3")
            log_path = os.path.join(tmpdir, "smc.log")
            best_thetas, best_scores, _ = wrapper_impjax.run_smc_sampling(
                self.built,
                self.sf,
                jax.random.PRNGKey(1),
                n_particles=10,
                n_temperature_steps=5,
                n_mcmc_steps=3,
                rmf_path=rmf_path,
                log_path=log_path,
                verbose=False,
            )
            handle = RMF.open_rmf_file_read_only(rmf_path)
            self.assertEqual(handle.get_number_of_frames(), len(best_thetas))

            stat_path = os.path.join(tmpdir, "smc_stats.csv")
            with open(stat_path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows) - 1, len(best_thetas))

            with open(log_path) as f:
                log_contents = f.read()
            self.assertIn("run_smc_sampling starting", log_contents)
            self.assertIn("run_smc_sampling finished", log_contents)

            layout = dof_layout.build(self.built)
            theta_now = state_sync.extract(self.built, layout)
            np.testing.assert_allclose(
                theta_now["bead_coords"], best_thetas[-1]["bead_coords"], atol=1e-6
            )

    def test_debug_mode_writes_score_comparison_for_every_temperature_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "smc.log")
            best_thetas, _, _ = wrapper_impjax.run_smc_sampling(
                self.built,
                self.sf,
                jax.random.PRNGKey(2),
                n_particles=10,
                n_temperature_steps=4,
                n_mcmc_steps=3,
                log_path=log_path,
                debug=True,
                verbose=False,
            )
            comparison_path = f"{os.path.splitext(log_path)[0]}_score_comparison.csv"
            with open(comparison_path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows) - 1, len(best_thetas))
            abs_diffs = np.array([float(r[3]) for r in rows[1:]])
            # connectivity-only toy system: JAX and CPU-IMP should agree exactly.
            np.testing.assert_allclose(abs_diffs, 0.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
