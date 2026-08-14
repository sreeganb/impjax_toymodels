"""Unit tests for score_verification.py and the custom_rmh step_callback hook
it plugs into.

test_run_sampling_debug_mode_matches_ground_truth is the load-bearing check:
on the connectivity-only toy system, IMP's CPU score and its JAX export must
agree to floating-point precision at every checkpoint (they're supposed to
be the same restraint, computed two different ways).
"""

import csv
import os
import tempfile
import unittest

import jax
import numpy as np

from impjax_toymodels import dof_layout, score_verification, wrapper_impjax
from toy_fixture import build_toy_system


class ScoreComparisonWriterTests(unittest.TestCase):
    def test_record_writes_matching_score_and_zero_diff_for_true_ground_truth(self):
        built, sf = build_toy_system()
        sf.evaluate(False)
        layout = dof_layout.build(built)
        context = wrapper_impjax.build_log_prob(built, sf)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "compare.csv")
            with score_verification.ScoreComparisonWriter(csv_path) as writer:
                log_prob = float(context.log_prob_fn(context.initial_theta))
                diff = writer.record(0, context.initial_theta, layout, built, sf, log_prob)

            self.assertAlmostEqual(diff, 0.0, places=4)
            with open(csv_path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["step", "jax_score", "imp_score", "abs_diff"])
            self.assertEqual(len(rows), 2)


class DebugModeIntegrationTests(unittest.TestCase):
    def test_run_sampling_debug_mode_matches_ground_truth(self):
        built, sf = build_toy_system()
        sf.evaluate(False)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "run.log")
            wrapper_impjax.run_sampling(
                built,
                sf,
                jax.random.PRNGKey(0),
                n_steps=20,
                mode="all",
                sigma_rotation=0.02,
                sigma_translation=0.5,
                sigma_bead=0.5,
                debug=True,
                debug_every=4,
                log_path=log_path,
                verbose=False,
            )
            comparison_path = f"{os.path.splitext(log_path)[0]}_score_comparison.csv"
            with open(comparison_path, newline="") as f:
                rows = list(csv.reader(f))

            # steps 0, 4, 8, 12, 16 -> 5 checkpoints + header
            self.assertEqual(len(rows) - 1, 5)
            abs_diffs = np.array([float(r[3]) for r in rows[1:]])
            np.testing.assert_allclose(abs_diffs, 0.0, atol=1e-3)

    def test_debug_requires_a_path_to_derive_the_comparison_csv_from(self):
        built, sf = build_toy_system()
        sf.evaluate(False)
        with self.assertRaises(ValueError):
            wrapper_impjax.run_sampling(
                built,
                sf,
                jax.random.PRNGKey(0),
                n_steps=5,
                debug=True,
                verbose=False,
            )

    def test_debug_false_produces_no_comparison_file(self):
        built, sf = build_toy_system()
        sf.evaluate(False)
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "run.log")
            wrapper_impjax.run_sampling(
                built, sf, jax.random.PRNGKey(0), n_steps=5, debug=False, log_path=log_path, verbose=False
            )
            comparison_path = f"{os.path.splitext(log_path)[0]}_score_comparison.csv"
            self.assertFalse(os.path.exists(comparison_path))


if __name__ == "__main__":
    unittest.main()
