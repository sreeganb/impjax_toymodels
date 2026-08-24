"""Tests for CSV-driven harmonic distance restraints.

Two layers are covered.  The first is pure file handling -- parsing, copy
wildcard expansion, round-tripping, and the error messages a malformed
constraint file has to produce -- and needs no IMP system at all.  The second
attaches the real KCOIL/ECOIL constraint file to the real system and checks
the property that makes the whole exercise meaningful: the restraints derived
from the ground-truth structure score (almost exactly) zero when the system is
put back into that structure, and score enormously when it is shuffled away.
"""

import os
import sys
import tempfile
import unittest

import IMP
import IMP.atom
import IMP.pmi.restraints.basic
import IMP.pmi.restraints.stereochemistry
import numpy as np

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

CONSTRAINT_FILE = os.path.join(EXAMPLES_DIR, "data", "distance_constraints.csv")
#: Residue range represented by flexible beads, i.e. with no reference position.
LINKER = (22, 31)

import evaluate_recovery  # noqa: E402
import kcoil_ecoil_system  # noqa: E402

from impjax_toymodels import distance_restraints as dr  # noqa: E402

HEADER = ",".join(dr.COLUMNS)


def write_csv(rows):
    """Write a throwaway constraint file and return its path."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    handle.write(HEADER + "\n")
    for row in rows:
        handle.write(row + "\n")
    handle.close()
    return handle.name


class ConstraintFileTests(unittest.TestCase):
    def test_reads_columns_in_any_order(self):
        # DictReader keys by name, so a file whose columns are permuted must
        # still parse identically -- otherwise hand-edited files break subtly.
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        handle.write("protein1,residue1,copy1,protein2,residue2,copy2,force_constant,distance\n")
        handle.write("KCOIL,5,0,ECOIL,6,1,2.5,7.25\n")
        handle.close()

        (constraint,) = dr.read_distance_constraints(handle.name)
        self.assertEqual(constraint.protein1, "KCOIL")
        self.assertEqual(constraint.residue1, 5)
        self.assertEqual(constraint.copy1, 0)
        self.assertEqual(constraint.copy2, 1)
        self.assertAlmostEqual(constraint.distance, 7.25)
        self.assertAlmostEqual(constraint.force_constant, 2.5)

    def test_round_trip_is_lossless(self):
        path = write_csv(["5,KCOIL,0,6,ECOIL,0,7.250,1.000"])
        original = dr.read_distance_constraints(path)
        out = os.path.join(tempfile.mkdtemp(), "out.csv")
        dr.write_distance_constraints(out, original)
        self.assertEqual(original, dr.read_distance_constraints(out))

    def test_wildcard_expands_copy_for_copy(self):
        path = write_csv([f"5,KCOIL,{dr.ANY_COPY},6,ECOIL,{dr.ANY_COPY},7.250,1.000"])
        expanded = dr.expand_copies(dr.read_distance_constraints(path), copy_number=3)

        # One restraint per copy, each pairing copy i with copy i -- never
        # copy i of one protein with copy j of the other, which would be a
        # different (cross-assembly) claim.
        self.assertEqual([(c.copy1, c.copy2) for c in expanded], [(0, 0), (1, 1), (2, 2)])
        self.assertTrue(all(c.distance == 7.25 for c in expanded))

    def test_explicit_copies_pass_through(self):
        path = write_csv(["5,KCOIL,0,6,ECOIL,1,7.250,1.000"])
        expanded = dr.expand_copies(dr.read_distance_constraints(path), copy_number=2)
        self.assertEqual([(c.copy1, c.copy2) for c in expanded], [(0, 1)])

    def test_copy_index_beyond_copy_number_is_an_error(self):
        # Silently dropping (or worse, mis-selecting) such a row would give a
        # system that is quietly under-restrained.
        path = write_csv(["5,KCOIL,3,6,ECOIL,3,7.250,1.000"])
        with self.assertRaisesRegex(ValueError, "only 2 copies"):
            dr.expand_copies(dr.read_distance_constraints(path), copy_number=2)

    def test_mixed_wildcard_and_index_is_an_error(self):
        path = write_csv([f"5,KCOIL,{dr.ANY_COPY},6,ECOIL,0,7.250,1.000"])
        with self.assertRaisesRegex(ValueError, "both copy columns"):
            dr.read_distance_constraints(path)

    def test_missing_column_is_reported_by_name(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        handle.write("residue1,protein1,copy1,residue2,protein2,copy2,distance\n")
        handle.write("5,KCOIL,0,6,ECOIL,0,7.250\n")
        handle.close()
        with self.assertRaisesRegex(ValueError, "force_constant"):
            dr.read_distance_constraints(handle.name)

    def test_non_positive_force_constant_is_an_error(self):
        path = write_csv(["5,KCOIL,0,6,ECOIL,0,7.250,0.000"])
        with self.assertRaisesRegex(ValueError, "force_constant must be > 0"):
            dr.read_distance_constraints(path)

    def test_bad_copy_value_names_the_row(self):
        path = write_csv(["5,KCOIL,0,6,ECOIL,0,7.250,1.000",
                          "5,KCOIL,two,6,ECOIL,0,7.250,1.000"])
        with self.assertRaisesRegex(ValueError, "row 3"):
            dr.read_distance_constraints(path)

    def test_blank_lines_are_tolerated(self):
        path = write_csv(["5,KCOIL,0,6,ECOIL,0,7.250,1.000", ",,,,,,,"])
        self.assertEqual(len(dr.read_distance_constraints(path)), 1)


class GroundTruthRecoveryTests(unittest.TestCase):
    """The property the generated constraint file exists to have."""

    @staticmethod
    def distance_restraint_score(output_objects):
        restraints = [r for r in output_objects
                      if isinstance(r, IMP.pmi.restraints.basic.DistanceRestraint)]
        return len(restraints), sum(r.get_restraint().get_score() for r in restraints)

    def test_reference_state_sits_at_the_restraint_minimum(self):
        # Only the restraints between structured beads can be checked this
        # way: `shuffle=False` puts the rigid bodies on their PDB coordinates,
        # but the flexible linker has no structure read in, so PMI leaves its
        # beads on a placeholder and nothing repositions them.  Restraints
        # touching a linker bead are therefore expected to be far from their
        # minimum here, which is exactly the freedom sampling has to resolve.
        built, _, output_objects = kcoil_ecoil_system.build_kcoil_ecoil_system(
            copy_number=1, shuffle=False)
        restraints = [r for r in output_objects
                      if isinstance(r, IMP.pmi.restraints.basic.DistanceRestraint)]
        self.assertGreater(len(restraints), 0)

        constraints = dr.expand_copies(dr.read_distance_constraints(CONSTRAINT_FILE), 1)
        self.assertEqual(len(constraints), len(restraints))
        structured = [
            restraint.get_restraint().get_score()
            for constraint, restraint in zip(constraints, restraints)
            if not (LINKER[0] <= constraint.residue1 <= LINKER[1]
                    or LINKER[0] <= constraint.residue2 <= LINKER[1])]
        self.assertGreater(len(structured), 0)
        # These targets were measured in exactly this state, so all that is
        # left is float round-trip noise.
        self.assertLess(sum(structured), 1e-3)

    def test_restraint_set_stays_sparse(self):
        # Guard against the constraint file drifting back toward a contact
        # map: every restraint is a separate node in the exported JAX graph,
        # so the count is a direct cost, and sparse data is the point.
        constraints = dr.read_distance_constraints(CONSTRAINT_FILE)
        self.assertLessEqual(len(constraints), 40)

    def test_shuffled_state_is_heavily_penalised(self):
        _, _, output_objects = kcoil_ecoil_system.build_kcoil_ecoil_system(
            copy_number=1, shuffle=True)
        _, score = self.distance_restraint_score(output_objects)
        self.assertGreater(score, 1e3)

    def test_restraint_count_scales_with_copy_number(self):
        _, _, one = kcoil_ecoil_system.build_kcoil_ecoil_system(copy_number=1, shuffle=False)
        _, _, three = kcoil_ecoil_system.build_kcoil_ecoil_system(copy_number=3, shuffle=False)
        count_one, _ = self.distance_restraint_score(one)
        count_three, _ = self.distance_restraint_score(three)
        self.assertEqual(count_three, 3 * count_one)

    def test_reference_state_has_zero_rmsd_to_itself(self):
        # Closes the loop between the two halves of the setup: the state the
        # restraints were measured in is the same state evaluate_recovery.py
        # compares sampled models against.  If these ever drift apart, every
        # reported RMSD is offset by a constant and nobody notices.
        reference = evaluate_recovery.reference_coordinates(EXAMPLES_DIR)
        self.assertGreater(len(reference), 0)
        self.assertAlmostEqual(
            evaluate_recovery.superposed_rmsd(reference, reference), 0.0, places=6)

    def test_superposed_rmsd_ignores_rigid_motion_but_not_reflection(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(40, 3)) * 10.0
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1.0
        moved = (rotation @ points.T).T + np.array([100.0, -50.0, 7.0])
        self.assertAlmostEqual(
            evaluate_recovery.superposed_rmsd(moved, points), 0.0, places=6)
        # A mirror image is a different structure and must not score as a match.
        mirrored = points * np.array([1.0, 1.0, -1.0])
        self.assertGreater(evaluate_recovery.superposed_rmsd(mirrored, points), 1.0)

    def test_recovery_rmsd_covers_structured_beads_only(self):
        # The linker is genuinely flexible and has no reference conformation,
        # so scoring it would add noise to the RMSD without adding meaning.
        built, _, _ = kcoil_ecoil_system.build_kcoil_ecoil_system(
            copy_number=1, shuffle=False)
        scored = len(evaluate_recovery.bead_coordinates(built.root_hier, copy_index=0))
        everything = len(IMP.atom.Selection(
            built.root_hier, resolution=1).get_selected_particles())
        self.assertGreater(scored, 0)
        self.assertLess(scored, everything)

    def test_split_builder_scores_distance_restraints_as_likelihood(self):
        # Regression guard: these restraints used to be added to the IMP model
        # but left out of both scoring functions on the split path, so the
        # prior/likelihood sampler was inferring without any of the data.
        _, likelihood_sf, prior_sf, output_objects = \
            kcoil_ecoil_system.build_kcoil_ecoil_split(copy_number=1, shuffle=True)
        _, distance_score = self.distance_restraint_score(output_objects)
        (excluded_volume,) = [
            r for r in output_objects
            if isinstance(r, IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere)]

        # The likelihood is exactly excluded volume + the distance restraints:
        # asserting the sum (rather than just "bigger than") pins down both
        # that the restraints are in there and that nothing else crept in.
        self.assertGreater(distance_score, 0.0)
        self.assertAlmostEqual(
            likelihood_sf.evaluate(False),
            excluded_volume.get_restraint().get_score() + distance_score,
            delta=1e-6 * max(1.0, distance_score))
        # ...and the prior side is connectivity only.  Compared against the
        # connectivity objects directly rather than against the distance score:
        # which of the two is larger depends on the configuration and on how
        # sparse the constraint file is, so it says nothing about the split.
        connectivity = sum(
            r.get_restraint().get_score() for r in output_objects
            if isinstance(r, IMP.pmi.restraints.stereochemistry.ConnectivityRestraint))
        self.assertAlmostEqual(prior_sf.evaluate(False), connectivity,
                               delta=1e-6 * max(1.0, connectivity))


if __name__ == "__main__":
    unittest.main()
