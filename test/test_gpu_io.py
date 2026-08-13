"""Unit tests for gpu_io.py: batched RMF3 + stat-file writing.

Verifies both the low-level claim the module relies on (repeated write_rmf
calls on one open handle append frames rather than overwrite) and the
higher-level contract (write_block syncs each state into IMP before
recording it, and produces a stat file whose rows match the log_probs
handed in).
"""

import csv
import os
import tempfile
import unittest

import jax
import numpy as np
import RMF

from impjax_toymodels import dof_layout, gpu_io, state_sync, wrapper_impjax
from toy_fixture import build_toy_system


class GpuIoTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.sf.evaluate(False)
        self.layout = dof_layout.build(self.built)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.rmf_path = os.path.join(self.tmpdir.name, "traj.rmf3")
        self.stat_path = os.path.join(self.tmpdir.name, "stats.csv")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_write_frame_appends_rather_than_overwrites(self):
        writer = gpu_io.TrajectoryWriter(self.rmf_path, self.stat_path, self.built.root_hier)
        writer.write_frame(0, -1.0)
        writer.write_frame(1, -2.0)
        writer.write_frame(2, -3.0)
        writer.close()

        handle = RMF.open_rmf_file_read_only(self.rmf_path)
        self.assertEqual(handle.get_number_of_frames(), 3)

    def test_write_block_syncs_states_and_records_matching_stat_rows(self):
        positions, log_probs, _ = wrapper_impjax.run_sampling(
            self.built,
            self.sf,
            jax.random.PRNGKey(0),
            n_steps=10,
            mode="beads",
            sigma_bead=2.0,
            sync_back=False,
            verbose=False,
        )

        with gpu_io.TrajectoryWriter(self.rmf_path, self.stat_path, self.built.root_hier) as writer:
            gpu_io.write_block(writer, positions, log_probs, self.layout, self.built)
            self.assertEqual(writer.n_frames, len(positions))

        handle = RMF.open_rmf_file_read_only(self.rmf_path)
        self.assertEqual(handle.get_number_of_frames(), len(positions))

        with open(self.stat_path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["step", "log_prob"])
        self.assertEqual(len(rows) - 1, len(positions))
        recorded_log_probs = np.array([float(r[1]) for r in rows[1:]])
        np.testing.assert_allclose(recorded_log_probs, log_probs, atol=1e-5)

        # After write_block, the IMP model reflects the *last* written state.
        theta_now = state_sync.extract(self.built, self.layout)
        np.testing.assert_allclose(theta_now["bead_coords"], positions[-1]["bead_coords"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
