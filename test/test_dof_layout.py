"""Unit tests for dof_layout.py against a real toy IMP system."""

import unittest

import numpy as np

from impjax_toymodels import dof_layout
from toy_fixture import build_toy_system


class DofLayoutTests(unittest.TestCase):
    def setUp(self):
        self.built, _ = build_toy_system()
        self.layout = dof_layout.build(self.built)

    def test_counts_match_built_system(self):
        rigid_bodies, beads = self.built.rigid_bodies_and_beads()
        self.assertEqual(self.layout.n_rigid_bodies, len(rigid_bodies))
        self.assertEqual(self.layout.n_beads, len(beads))
        self.assertEqual(self.layout.flat_size, 7 * len(rigid_bodies) + 3 * len(beads))

    def test_rigid_body_member_indexes_are_unique_and_disjoint_from_beads(self):
        member_indexes = np.concatenate(
            [rb.member_particle_indexes for rb in self.layout.rigid_bodies]
        )
        self.assertEqual(len(member_indexes), len(set(member_indexes.tolist())))
        self.assertTrue(
            set(member_indexes.tolist()).isdisjoint(self.layout.bead_particle_indexes.tolist())
        )

    def test_flatten_unflatten_roundtrip(self):
        rng = np.random.default_rng(0)
        theta = {
            "quaternions": rng.normal(size=(self.layout.n_rigid_bodies, 4)),
            "translations": rng.normal(size=(self.layout.n_rigid_bodies, 3)),
            "bead_coords": rng.normal(size=(self.layout.n_beads, 3)),
        }
        flat = dof_layout.flatten(theta, self.layout)
        self.assertEqual(flat.shape, (self.layout.flat_size,))
        restored = dof_layout.unflatten(flat, self.layout)
        for key in theta:
            np.testing.assert_allclose(restored[key], theta[key])

    def test_mode_mask_covers_all_named_modes(self):
        for mode in dof_layout.SAMPLING_MODES:
            mask = dof_layout.mode_mask(mode)
            self.assertTrue(mask.rotate_rigid or mask.translate_rigid or mask.translate_beads)
        with self.assertRaises(ValueError):
            dof_layout.mode_mask("not-a-mode")


if __name__ == "__main__":
    unittest.main()
