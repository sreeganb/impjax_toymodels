"""Tests for ground-truth contact maps and restraint selection.

The contact map is the durable artifact of the restraint pipeline -- generated
once per ground-truth structure and reused for every run -- so its round trip
and the selection rules reading it are worth pinning down precisely.
"""

import os
import sys
import tempfile
import unittest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from impjax_toymodels import contact_map as cm  # noqa: E402

MAP_FILE = os.path.join(EXAMPLES_DIR, "data", "contact_maps", "kcoil_ecoil_n1.csv")


def pair(rank=1, r1=1, p1="KCOIL", c1=0, t1="K", b1="KCOIL_0_1-21",
         r2=7, p2="ECOIL", c2=0, t2="K", b2="ECOIL_0_1-21", distance=10.0, kind=None):
    return cm.ContactPair(
        rank=rank, residue1=r1, protein1=p1, copy1=c1, restype1=t1, body1=b1,
        residue2=r2, protein2=p2, copy2=c2, restype2=t2, body2=b2,
        distance=distance, kind=kind or cm.classify(p1, c1, p2, c2))


class ContactMapFileTests(unittest.TestCase):
    def test_round_trip_is_lossless(self):
        original = [pair(rank=1, distance=4.5), pair(rank=2, r1=6, distance=9.25)]
        path = os.path.join(tempfile.mkdtemp(), "map.csv")
        cm.write_contact_map(path, original)
        self.assertEqual(original, cm.read_contact_map(path))

    def test_missing_column_is_reported_by_name(self):
        path = os.path.join(tempfile.mkdtemp(), "bad.csv")
        with open(path, "w") as handle:
            handle.write("rank,residue1\n1,5\n")
        with self.assertRaisesRegex(ValueError, "distance"):
            cm.read_contact_map(path)

    def test_classify_distinguishes_all_three_relationships(self):
        self.assertEqual(cm.classify("KCOIL", 0, "ECOIL", 0), cm.INTER_MOL)
        self.assertEqual(cm.classify("KCOIL", 0, "KCOIL", 0), cm.INTRA_MOL)
        self.assertEqual(cm.classify("KCOIL", 0, "ECOIL", 1), cm.INTER_COPY)
        # Copy always wins: a same-protein pair across copies is still a
        # statement about two assemblies, not about one chain.
        self.assertEqual(cm.classify("KCOIL", 0, "KCOIL", 1), cm.INTER_COPY)


class SelectionTests(unittest.TestCase):
    def test_same_rigid_body_pairs_are_dropped(self):
        # Their distance cannot change, so the restraint is a constant.
        frozen = pair(p2="KCOIL", r2=15, b2="KCOIL_0_1-21", distance=5.0)
        self.assertTrue(frozen.same_body)
        self.assertEqual(cm.select([frozen], top_n=10, top_n_intra=10), [])

    def test_flexible_pairs_are_kept(self):
        # A linker bead shares no rigid body, so it is never "frozen".
        flexible = pair(p2="KCOIL", r2=24, t2="S", b2=cm.FLEXIBLE, distance=5.0)
        self.assertTrue(flexible.is_flexible)
        self.assertFalse(flexible.same_body)
        self.assertEqual(len(cm.select([flexible], top_n=10, top_n_intra=10)), 1)

    def test_residue_type_filter_requires_both_ends(self):
        pairs = [pair(t1="K", t2="K", distance=5.0),
                 pairs_ala := pair(t1="K", t2="A", distance=4.0)]
        kept = cm.select(pairs, residue_types={"K", "S"}, top_n=10)
        self.assertEqual([p.restype2 for p in kept], ["K"])
        self.assertNotIn(pairs_ala, kept)

    def test_sequence_neighbours_are_dropped_within_a_chain(self):
        # Connectivity already restrains these.
        adjacent = pair(p2="KCOIL", r1=24, r2=25, b1=cm.FLEXIBLE, b2=cm.FLEXIBLE,
                        distance=3.0)
        self.assertEqual(cm.select([adjacent], min_seq_sep=3, top_n_intra=10), [])
        self.assertEqual(len(cm.select([adjacent], min_seq_sep=1, top_n_intra=10)), 1)

    def test_budget_is_per_group_not_global(self):
        # One tight interface must not consume the whole budget and leave
        # another part of the assembly unrestrained.
        tight = [pair(r1=r, distance=1.0 + 0.1 * r) for r in range(1, 8)]
        other = [pair(r1=r, c1=1, c2=1, distance=20.0 + r) for r in range(1, 8)]
        kept = cm.select(tight + other, top_n=3)
        by_copy = {}
        for p in kept:
            by_copy[p.copy1] = by_copy.get(p.copy1, 0) + 1
        self.assertEqual(by_copy, {0: 3, 1: 3})

    def test_bead_key_deduplicates_pairs_sharing_a_bead(self):
        # Residues 5 and 6 share a resolution-2 bead, so restraining both is
        # two near-identical restraints on one particle pair.
        same_bead = [pair(r1=5, distance=8.0), pair(r1=6, distance=9.0)]
        key = lambda p: (p.protein1, p.copy1, (5, 6), p.protein2, p.copy2, (7, 8))
        self.assertEqual(len(cm.select(same_bead, top_n=10, bead_key=key)), 1)
        # Without a key, both survive.
        self.assertEqual(len(cm.select(same_bead, top_n=10)), 2)
        # The closer one is the one kept.
        (kept,) = cm.select(same_bead, top_n=10, bead_key=key)
        self.assertAlmostEqual(kept.distance, 8.0)

    def test_results_are_ordered_closest_first(self):
        pairs = [pair(r1=r, distance=d) for r, d in ((1, 9.0), (6, 4.0), (8, 7.0))]
        self.assertEqual([p.distance for p in cm.select(pairs, top_n=10)],
                         [4.0, 7.0, 9.0])


class ShippedMapTests(unittest.TestCase):
    """The committed KCOIL/ECOIL map, as the pipeline actually uses it."""

    @classmethod
    def setUpClass(cls):
        cls.pairs = cm.read_contact_map(MAP_FILE)

    def test_map_is_ranked_closest_first(self):
        distances = [p.distance for p in self.pairs]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual([p.rank for p in self.pairs],
                         list(range(1, len(self.pairs) + 1)))

    def test_default_selection_is_sparse_and_spans_both_chains(self):
        kept = cm.select(self.pairs, residue_types={"K", "S"}, top_n=10, top_n_intra=2)
        self.assertLessEqual(len(kept), 20)
        self.assertTrue(any(p.kind == cm.INTER_MOL for p in kept))
        self.assertTrue(any(p.kind == cm.INTRA_MOL for p in kept))
        # Serine is in the default set precisely so the linker is not data-free.
        self.assertTrue(any(p.is_flexible for p in kept))
