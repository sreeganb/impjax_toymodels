"""Tests for the CAPRI-style docking metrics.

Two kinds of check:

* **Invariance and exactness.** Every metric must be blind to where the whole
  complex sits in space, and a known perturbation must produce the known
  answer -- a 5 A ligand translation is a 5.0 A ligand RMSD, not approximately
  one. That is what separates a metric from a number that merely goes up.
* **Agreement with IMP.** The global RMSD is cross-checked against
  `IMP.pmi.analysis.Precision`, which is the primitive PMI_analysis's
  accuracy.py wraps. The clustering pipeline around it is not used (it needs
  cluster.N.sample_A/B.txt files from a replica-exchange analysis run), but
  the underlying measurement is the same one, so this pins our Kabsch
  implementation against IMP's own.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

import IMP
import IMP.atom
import IMP.pmi.analysis
import IMP.pmi.tools
import IMP.rmf
import RMF

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
DOCKING_DIR = os.path.join(EXAMPLES_DIR, "rigid_body_docking")
for _directory in (DOCKING_DIR, os.path.join(EXAMPLES_DIR, "harness")):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import docking_metrics  # noqa: E402
import docking_system  # noqa: E402
import evaluate_recovery  # noqa: E402


def _rotation(angle: float) -> np.ndarray:
    """A proper rotation about z, for invariance checks."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


class DockingMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator, cls.reference = docking_metrics.build_evaluator(docking_system)

    def test_reference_scores_perfectly_against_itself(self):
        scores = self.evaluator.score(self.reference)
        self.assertLess(scores.rmsd, 1e-9)
        self.assertLess(scores.ligand_rmsd, 1e-9)
        self.assertLess(scores.interface_rmsd, 1e-9)
        self.assertEqual(scores.fnat, 1.0)
        self.assertEqual(scores.capri, "high")

    def test_every_metric_is_blind_to_where_the_complex_sits(self):
        """Nothing in the scoring function knows about the origin, so the
        recovered assembly is free to sit anywhere; a metric that noticed
        would be measuring that freedom instead of the structure."""
        moved = (_rotation(0.7) @ self.reference.T).T + np.array([25.0, -13.0, 7.0])
        scores = self.evaluator.score(moved)
        self.assertLess(scores.rmsd, 1e-9)
        self.assertLess(scores.ligand_rmsd, 1e-9)
        self.assertLess(scores.interface_rmsd, 1e-9)
        self.assertEqual(scores.fnat, 1.0)

    def test_ligand_translation_gives_exactly_that_ligand_rmsd(self):
        """Translating the ligand 5 A can only be a 5.0 A ligand RMSD: the
        receptor superposition is exact, so the whole displacement lands on
        the ligand."""
        moved = self.reference.copy()
        moved[self.evaluator.ligand] += np.array([5.0, 0.0, 0.0])
        scores = self.evaluator.score(moved)
        self.assertAlmostEqual(scores.ligand_rmsd, 5.0, places=9)

    def test_global_rmsd_understates_a_pure_ligand_error(self):
        """Why the docking metrics exist at all: a global superposition
        spreads the ligand's error across both partners, so it reports a much
        smaller number than the placement error actually is."""
        moved = self.reference.copy()
        moved[self.evaluator.ligand] += np.array([5.0, 0.0, 0.0])
        scores = self.evaluator.score(moved)
        self.assertLess(scores.rmsd, 0.5 * scores.ligand_rmsd)

    def test_fnat_degrades_as_the_ligand_moves_away(self):
        previous = 1.0
        for shift in (2.0, 5.0, 10.0, 30.0):
            moved = self.reference.copy()
            moved[self.evaluator.ligand] += np.array([shift, 0.0, 0.0])
            fraction = self.evaluator.score(moved).fnat
            with self.subTest(shift=shift):
                self.assertLessEqual(fraction, previous)
            previous = fraction
        self.assertEqual(previous, 0.0)  # 30 A apart: no native contact left

    def test_a_badly_placed_ligand_is_classed_incorrect(self):
        moved = self.reference.copy()
        moved[self.evaluator.ligand] += np.array([40.0, 0.0, 0.0])
        self.assertEqual(self.evaluator.score(moved).capri, "incorrect")

    def test_reference_defines_a_real_interface(self):
        """The interface and contact sets must be non-trivial, or fnat and
        I-RMSD would be measuring almost nothing."""
        self.assertGreater(int(self.evaluator.contacts.sum()), 20)
        self.assertGreater(len(self.evaluator.interface), 20)
        # ...and must not have swallowed the whole structure either.
        self.assertLess(len(self.evaluator.interface), len(self.reference))

    def test_row_slices_partition_the_bead_array(self):
        receptor, ligand = self.evaluator.receptor, self.evaluator.ligand
        self.assertEqual(receptor.start, 0)
        self.assertEqual(receptor.stop, ligand.start)
        self.assertEqual(ligand.stop, len(self.reference))


class CapriClassTests(unittest.TestCase):
    """The published thresholds, spelled out so a refactor cannot drift."""

    def test_bands(self):
        classify = docking_metrics.capri_class
        self.assertEqual(classify(0.6, 0.8, 0.5), "high")
        self.assertEqual(classify(0.4, 4.0, 1.5), "medium")
        self.assertEqual(classify(0.2, 9.0, 3.5), "acceptable")
        self.assertEqual(classify(0.05, 9.0, 3.5), "incorrect")
        # fnat alone is never enough, and neither is RMSD alone.
        self.assertEqual(classify(0.9, 40.0, 40.0), "incorrect")
        self.assertEqual(classify(0.0, 0.1, 0.1), "incorrect")


class ImpAgreementTests(unittest.TestCase):
    """Our Kabsch RMSD against IMP's own, on real RMF3 files."""

    def test_matches_imp_pmi_analysis_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = os.path.join(directory, "reference.rmf3")
            trajectory_path = os.path.join(directory, "trajectory.rmf3")

            reference = _write_frames(reference_path, shuffle=False, n_frames=1)[0]
            frames = _write_frames(trajectory_path, shuffle=True, n_frames=3)

            ours = [evaluate_recovery.superposed_rmsd(frame, reference)
                    for frame in frames]

            model = IMP.Model()
            precision = IMP.pmi.analysis.Precision(
                model, resolution=1,
                selection_dictionary={"selection": list(docking_system.PROTEINS)})
            precision.set_precision_style("pairwise_rmsd")
            precision.add_structures(
                [(trajectory_path, frame) for frame in range(3)], "set0")
            precision.set_reference_structure(reference_path, 0)
            theirs = precision.get_rmsd_wrt_reference_structure_with_alignment(
                "set0", "selection")["selection"]["all_distances"]

        self.assertEqual(len(ours), len(theirs))
        for mine, imp in zip(ours, theirs):
            # Both are genuinely large here (shuffled starts), so agreement to
            # 1e-4 A is agreement on the measurement, not on a shared zero.
            self.assertGreater(mine, 1.0)
            self.assertAlmostEqual(mine, imp, places=4)


def _write_frames(path: str, shuffle: bool, n_frames: int):
    """Write an RMF3 trajectory, returning each frame's bead coordinates."""
    built, _, _ = docking_system.build_system(shuffle=shuffle, distance_csv=False)
    handle = RMF.create_rmf_file(path)
    IMP.rmf.add_hierarchy(handle, built.root_hier)

    molecules = [mol for mols in built.molecules.values() for mol in mols]
    coordinates = []
    for frame in range(n_frames):
        IMP.rmf.save_frame(handle, str(frame))
        coordinates.append(evaluate_recovery.bead_coordinates(
            built.root_hier, copy_index=0, system=docking_system))
        if frame < n_frames - 1:
            IMP.pmi.tools.shuffle_configuration(
                molecules, max_translation=8, avoidcollision_rb=False)
    del handle  # flush before the file is read back
    return coordinates


if __name__ == "__main__":
    unittest.main()
