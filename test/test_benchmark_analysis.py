"""Tests for the benchmark's accuracy measurement and its replica-exchange launcher.

Covers examples/harness/trajectory_analysis.py (best-scoring models, their
RMSD, the convergence trace, time to solution) and examples/harness/
run_imp_rex.py (the MPI command it builds, its refusal to run many ranks
without IMP.mpi, and a one-replica in-process run). No MPI job is started.
"""

import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import IMP
import IMP.pmi.output
import IMP.pmi.tools
import RMF

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
HARNESS_DIR = os.path.join(EXAMPLES_DIR, "harness")
for _directory in (EXAMPLES_DIR, HARNESS_DIR):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import kcoil_ecoil_system  # noqa: E402
import run_imp_rex  # noqa: E402
import structure_rmsd  # noqa: E402
import trajectory_analysis  # noqa: E402


class GlobalTraceTests(unittest.TestCase):
    def test_keeps_only_new_minima_in_time_order(self):
        # Two files' own improvement events, interleaved in time.
        events = [(1.0, 50.0, 9.0), (3.0, 20.0, 4.0), (2.0, 30.0, 6.0), (4.0, 25.0, 5.0)]
        times, scores, rmsds = trajectory_analysis._global_trace(events)
        self.assertEqual(times, [1.0, 2.0, 3.0])
        self.assertEqual(scores, [50.0, 30.0, 20.0])
        self.assertEqual(rmsds, [9.0, 6.0, 4.0])

    def test_frame_times_span_the_file(self):
        trajectory = trajectory_analysis.TrajectoryFile("x.rmf3", 10.0, 20.0)
        self.assertAlmostEqual(trajectory.time_of(0, 4), 12.5)
        self.assertAlmostEqual(trajectory.time_of(3, 4), 20.0)


class AnalyseRunTests(unittest.TestCase):
    """Three shuffled frames then the ground truth itself, in one RMF3 file."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.rmf_path = os.path.join(cls.tmpdir.name, "run.rmf3")
        IMP.random_number_generator.seed(3)
        built, _, _ = kcoil_ecoil_system.build_kcoil_ecoil_system(
            copy_number=1, shuffle=False, distance_csv=False)
        cls.reference = structure_rmsd.build_reference(kcoil_ecoil_system, 1, False)
        truth = {rb: rb.get_reference_frame() for rb in built.dof.get_rigid_bodies()}
        flexible = {p: IMP.core.XYZ(p).get_coordinates() for p in built.dof.get_flexible_beads()}

        output = IMP.pmi.output.Output()
        output.init_rmf(cls.rmf_path, [built.root_hier])
        molecules = [m for ms in built.molecules.values() for m in ms]
        for _ in range(3):
            IMP.pmi.tools.shuffle_configuration(molecules, max_translation=100)
            output.write_rmf(cls.rmf_path)
        for rb, frame in truth.items():
            rb.set_reference_frame(frame)
        for particle, xyz in flexible.items():
            IMP.core.XYZ(particle).set_coordinates(xyz)
        output.write_rmf(cls.rmf_path)
        output.close_rmf(cls.rmf_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def analyse(self, **kwargs):
        options = {"n_best": 2, "burnin_fraction": 0.0, "success_rmsd": 1.0, **kwargs}
        return trajectory_analysis.analyse_run(
            kcoil_ecoil_system, [trajectory_analysis.TrajectoryFile(self.rmf_path, 0.0, 8.0)],
            1, False, self.reference, **options)

    def test_best_scoring_model_is_the_ground_truth(self):
        result = self.analyse()
        self.assertEqual(result["n_frames"], 4)
        self.assertEqual(len(result["best_rmsds"]), 2)
        self.assertAlmostEqual(result["best_rmsds"][0], 0.0, places=4)
        self.assertAlmostEqual(result["best_scores"][0], self.reference.imp_score, places=4)
        self.assertTrue(np.all(np.diff(result["best_scores"]) >= 0))

    def test_time_to_solution_is_when_the_truth_appears(self):
        result = self.analyse()
        # Frame 3 of 4 in a file spanning 0-8 s is placed at 8 s.
        self.assertAlmostEqual(result["time_to_solution"], 8.0)
        self.assertAlmostEqual(result["trace_rmsd"][-1], 0.0, places=4)
        self.assertTrue(np.all(np.diff(result["trace_score"]) < 0))

    def test_burnin_excludes_early_frames_from_the_pool_only(self):
        result = self.analyse(n_best=10, burnin_fraction=0.5)
        self.assertEqual(len(result["best_rmsds"]), 2)   # frames 2 and 3 only
        self.assertEqual(result["n_frames"], 4)           # but all four were scored

    def test_an_unreached_threshold_reports_nan(self):
        result = trajectory_analysis.analyse_run(
            kcoil_ecoil_system, [trajectory_analysis.TrajectoryFile(self.rmf_path, 0.0, 8.0)],
            1, False, structure_rmsd.Reference(self.reference.coordinates + 0.0, 0.0),
            n_best=1, burnin_fraction=0.0, success_rmsd=-1.0)
        self.assertTrue(np.isnan(result["time_to_solution"]))


def _rex_args(**overrides):
    values = {"system": "kcoil_ecoil_system", "copy_number": 1, "distance_csv": False,
              "seed": 4, "imp_rex_frames": 3, "imp_rex_mc_steps": 2, "imp_rex_max_temp": 2.0,
              "imp_rex_replicas": 1, "imp_rex_launcher": "mpirun"}
    values.update(overrides)
    return argparse.Namespace(**values)


class ImpRexLauncherTests(unittest.TestCase):
    def test_command_runs_this_script_with_the_case_settings(self):
        argv = run_imp_rex.command(_rex_args(distance_csv="d.csv"), "out/rex")
        self.assertEqual(argv[:2], [sys.executable, run_imp_rex.HERE])
        for flag, value in (("--seed", "4"), ("--frames", "3"), ("--distance-csv", "d.csv"),
                            ("--output-dir", "out/rex")):
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_launched_ranks_reads_the_launcher_environment(self):
        with mock.patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "8"}):
            self.assertEqual(run_imp_rex.launched_ranks(), 8)

    def test_many_ranks_without_imp_mpi_is_refused(self):
        try:
            import IMP.mpi  # noqa: F401
            self.skipTest("this IMP has IMP.mpi")
        except ImportError:
            pass
        with mock.patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "4"}):
            with self.assertRaises(RuntimeError):
                run_imp_rex.replica_exchange_object()

    def test_single_replica_runs_in_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "rex")
            timing = run_imp_rex.launch(_rex_args(), output_dir)
            self.assertEqual(timing["replicas"], 1)
            self.assertGreater(timing["wall_time"], 0.0)
            handle = RMF.open_rmf_file_read_only(os.path.join(output_dir, "rmfs", "0.rmf3"))
            self.assertEqual(handle.get_number_of_frames(), 3)


if __name__ == "__main__":
    unittest.main()
