"""Tests for the 1AVX preparation step and the inputs it produced.

Two halves, deliberately separate:

* The pure slot-manipulation functions are tested against hand-built inputs,
  so break detection and domain derivation are checked without needing any
  particular structure file.
* The *committed* prepared inputs are then checked for the properties the
  whole example depends on -- above all that IMP reads each chain with no
  duplicate residue indexes, which is the failure the preparation exists to
  prevent and which is silent if it regresses.

The raw 1AVX download is not committed (see prepare_1avx.py), so nothing here
re-runs the fetch; the prepared outputs are the fixtures.
"""

import os
import sys
import unittest

import numpy as np

import IMP
import IMP.atom

DOCKING_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "rigid_body_docking")
if DOCKING_DIR not in sys.path:
    sys.path.insert(0, DOCKING_DIR)

import prepare_1avx  # noqa: E402

DATA_DIR = os.path.join(DOCKING_DIR, "data")

#: What prepare_1avx.py produces from the deposited entry, by default.
EXPECTED_RESIDUES = {"TRYP": 223, "STI": 172}


def _slot(restype: str, resseq: int, coordinates=None) -> prepare_1avx.Slot:
    """A minimal observed slot carrying only the backbone atoms breaks need."""
    slot = prepare_1avx.Slot(restype=restype, original=(resseq, ""))
    if coordinates is not None:
        for name, (x, y, z) in coordinates.items():
            slot.lines.append(
                f"ATOM      1  {name:<3s} {restype} A{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
    return slot


class SlotLogicTests(unittest.TestCase):
    """The numbering-independent logic, on inputs small enough to reason about."""

    def test_numbering_jump_without_a_broken_bond_is_not_a_break(self):
        # Exactly trypsin's situation: the residue number jumps by three while
        # the peptide bond is intact.  Treating this as a gap would insert
        # residues that are physically present.
        first = _slot("ALA", 34, {"C": (0.0, 0.0, 0.0)})
        second = _slot("GLY", 37, {"N": (1.33, 0.0, 0.0)})
        self.assertEqual(prepare_1avx.find_breaks([first, second]), [])

    def test_stretched_bond_is_a_break_even_with_consecutive_numbering(self):
        first = _slot("ALA", 10, {"C": (0.0, 0.0, 0.0)})
        second = _slot("GLY", 11, {"N": (11.41, 0.0, 0.0)})
        self.assertEqual(prepare_1avx.find_breaks([first, second]), [0])

    def test_structured_ranges_split_on_unobserved_slots(self):
        slots = [_slot("ALA", i) for i in range(1, 4)]
        slots.append(prepare_1avx.Slot(restype="GLN", original=None))
        slots.extend(_slot("GLY", i) for i in range(5, 8))
        self.assertEqual(prepare_1avx.structured_ranges(slots), [(1, 3), (5, 7)])

    def test_structured_ranges_of_a_fully_observed_chain_is_one_domain(self):
        slots = [_slot("ALA", i) for i in range(1, 21)]
        self.assertEqual(prepare_1avx.structured_ranges(slots), [(1, 20)])

    def test_insert_missing_fills_only_between_the_flanking_residues(self):
        slots = [_slot("ALA", 638, {"C": (0.0, 0.0, 0.0)}),
                 _slot("GLY", 644, {"N": (11.41, 0.0, 0.0)})]
        missing = [(639, "GLN"), (640, "ALA"), (641, "GLU"), (642, "ASP"),
                   (643, "ASP"), (700, "SER")]  # 700 lies outside the break
        filled = prepare_1avx.insert_missing(slots, [0], missing)
        self.assertEqual(len(filled), 7)
        self.assertEqual([s.observed for s in filled],
                         [True, False, False, False, False, False, True])


class PreparedInputTests(unittest.TestCase):
    """Properties of the committed prepared inputs."""

    def test_imp_reads_each_chain_without_duplicate_residue_indexes(self):
        # The reason this preparation step exists: the deposited chain A has
        # insertion codes 184A/188A/221A, and IMP drops the code, so reading
        # it directly yields 224 residues with 3 duplicated indexes.  PMI keys
        # on that index, so a duplicate corrupts the build with no error.
        for protein, expected in EXPECTED_RESIDUES.items():
            with self.subTest(protein=protein):
                model = IMP.Model()
                hierarchy = IMP.atom.read_pdb(
                    os.path.join(DATA_DIR, "pdb", f"{protein}.pdb"), model,
                    IMP.atom.NonWaterNonHydrogenPDBSelector())
                residues = IMP.atom.get_by_type(hierarchy, IMP.atom.RESIDUE_TYPE)
                indexes = [IMP.atom.Residue(r).get_index() for r in residues]
                self.assertEqual(len(residues), expected)
                self.assertEqual(len(set(indexes)), len(indexes))
                self.assertEqual(sorted(indexes), list(range(1, expected + 1)))

    def test_fasta_length_matches_the_residue_count(self):
        for protein, expected in EXPECTED_RESIDUES.items():
            with self.subTest(protein=protein):
                with open(os.path.join(DATA_DIR, "fasta", f"{protein}.fasta")) as handle:
                    header = handle.readline()
                    sequence = handle.read().replace("\n", "")
                # IMP.pmi.topology.Sequences keys on the "sp|" header form.
                self.assertTrue(header.startswith(">sp|"))
                self.assertEqual(len(sequence), expected)

    def test_domains_cover_every_residue_as_one_rigid_body(self):
        import json
        for protein, expected in EXPECTED_RESIDUES.items():
            with self.subTest(protein=protein):
                with open(os.path.join(DATA_DIR, "json_files", f"{protein}.json")) as handle:
                    info = json.load(handle)
                # The default preparation splices gaps out, so each chain is a
                # single structured domain -- which is what makes this a pure
                # two-rigid-body docking problem with no flexible beads.
                self.assertEqual(info["domains"], [[1, expected]])

    def test_complex_holds_both_chains_in_the_per_chain_frame(self):
        # The complex file is the ground truth, so its coordinates must be the
        # crystal's, identical to the per-chain files rather than re-derived.
        for protein, chain in (("TRYP", "A"), ("STI", "B")):
            with self.subTest(protein=protein):
                single = _atom_coordinates(
                    os.path.join(DATA_DIR, "pdb", f"{protein}.pdb"))
                combined = _atom_coordinates(
                    os.path.join(DATA_DIR, "pdb", "1avx_complex.pdb"), chain)
                np.testing.assert_array_equal(single, combined)

    def test_mapping_round_trips_to_the_deposited_numbering(self):
        import csv
        with open(os.path.join(DATA_DIR, "residue_mapping.csv")) as handle:
            rows = list(csv.DictReader(handle))

        for protein, expected in EXPECTED_RESIDUES.items():
            entries = [r for r in rows if r["protein"] == protein]
            with self.subTest(protein=protein):
                self.assertEqual(len(entries), expected)
                self.assertEqual([int(r["new_index"]) for r in entries],
                                 list(range(1, expected + 1)))
                self.assertTrue(all(r["observed"] == "1" for r in entries))

        # The three insertion-coded residues must survive as distinct rows --
        # they are exactly what collided before renumbering.
        coded = [r for r in rows if r["original_icode"]]
        self.assertEqual(sorted(r["original_resseq"] for r in coded),
                         ["184", "188", "221"])


def _atom_coordinates(path: str, chain: str = None) -> np.ndarray:
    """Every ATOM coordinate in a file, optionally restricted to one chain."""
    coordinates = []
    for line in open(path):
        if line.startswith("ATOM") and (chain is None or line[21] == chain):
            coordinates.append(
                [float(line[30 + 8 * i: 38 + 8 * i]) for i in range(3)])
    return np.asarray(coordinates)


if __name__ == "__main__":
    unittest.main()
