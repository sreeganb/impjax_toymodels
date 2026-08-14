"""End-to-end smoke test for the examples/ KCOIL/ECOIL pipeline.

This exercises the entire, real, non-negotiable pipeline the user asked
for: build the actual KCOIL/ECOIL system (examples/kcoil_ecoil_system.py,
not the minimal test/toy_fixture.py used by the other module-level tests),
sample it through wrapper_impjax.run_sampling, and check the RMF3
trajectory, stat file, and log file it produces are all consistent with
each other and with what the sampler actually did.

examples/ is not an importable package (by design -- it holds runnable,
system-specific scripts, not source code), so its directory is added to
sys.path here, the same way a user running the example script would have
it on their path by virtue of running python from within examples/.
"""

import csv
import os
import sys
import tempfile
import unittest

import jax
import numpy as np
import RMF

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from kcoil_ecoil_system import build_kcoil_ecoil_system  # noqa: E402

from impjax_toymodels import dof_layout, state_sync, wrapper_impjax  # noqa: E402


class KcoilEcoilExampleTests(unittest.TestCase):
    def test_build_kcoil_ecoil_system_scales_with_copy_number(self):
        built1, sf1, _ = build_kcoil_ecoil_system(copy_number=1)
        built2, sf2, _ = build_kcoil_ecoil_system(copy_number=2)

        layout1 = dof_layout.build(built1)
        layout2 = dof_layout.build(built2)

        # 2 domains (rigid bodies) x 2 proteins (KCOIL, ECOIL) x copy_number
        self.assertEqual(layout1.n_rigid_bodies, 4)
        self.assertEqual(layout2.n_rigid_bodies, 8)
        self.assertEqual(layout2.n_beads, 2 * layout1.n_beads)

        self.assertTrue(np.isfinite(sf1.evaluate(False)))
        self.assertTrue(np.isfinite(sf2.evaluate(False)))

    def test_full_pipeline_produces_consistent_rmf3_stat_and_log(self):
        built, score_function, _ = build_kcoil_ecoil_system(copy_number=1)
        score_function.evaluate(False)  # materialize IMP's JAX export

        with tempfile.TemporaryDirectory() as tmpdir:
            rmf_path = os.path.join(tmpdir, "kcoil_ecoil.rmf3")
            log_path = os.path.join(tmpdir, "kcoil_ecoil.log")

            positions, log_probs, acceptance_rate = wrapper_impjax.run_sampling(
                built,
                score_function,
                jax.random.PRNGKey(0),
                n_steps=15,
                mode="all",
                sigma_rotation=0.02,
                sigma_translation=0.5,
                sigma_bead=0.5,
                rmf_path=rmf_path,
                log_path=log_path,
                verbose=False,
            )

            self.assertEqual(len(positions), 15)
            self.assertTrue(np.all(np.isfinite(log_probs)))

            handle = RMF.open_rmf_file_read_only(rmf_path)
            self.assertEqual(handle.get_number_of_frames(), len(positions))

            stat_path = os.path.join(tmpdir, "kcoil_ecoil_stats.csv")
            with open(stat_path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows) - 1, len(positions))
            recorded_log_probs = np.array([float(r[1]) for r in rows[1:]])
            np.testing.assert_allclose(recorded_log_probs, log_probs, atol=1e-4)

            with open(log_path) as f:
                log_contents = f.read()
            self.assertIn("run_sampling starting", log_contents)
            self.assertIn("acceptance_rate=", log_contents)
            self.assertIn(f"samples_saved={len(positions)}", log_contents)

            layout = dof_layout.build(built)
            theta_now = state_sync.extract(built, layout)
            np.testing.assert_allclose(
                theta_now["bead_coords"], positions[-1]["bead_coords"], atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
