"""Unit tests for state_sync.py: the IMP <-> reduced-state bridge.

These are the tests that matter most for correctness of the whole wrapper:
they check that the expansion map Phi (doc/design.tex Section 4) produces
exactly the per-particle array IMP's own JAX-exported score function expects,
by comparing against IMP's own `evaluate()` on the same configuration.
"""

import unittest

import numpy as np

from impjax_toymodels import dof_layout, state_sync
from toy_fixture import build_toy_system

import IMP
import IMP.algebra
import IMP.core


class StateSyncTests(unittest.TestCase):
    def setUp(self):
        self.built, self.sf = build_toy_system()
        self.layout = dof_layout.build(self.built)
        self.ji = self.sf._get_jax()

    def test_extract_apply_roundtrip(self):
        theta = state_sync.extract(self.built, self.layout)
        state_sync.apply(theta, self.layout, self.built)
        theta_again = state_sync.extract(self.built, self.layout)
        for key in theta:
            np.testing.assert_allclose(theta_again[key], theta[key], atol=1e-8)

    def test_apply_moves_rigid_body_and_beads_as_expected(self):
        theta = state_sync.extract(self.built, self.layout)
        theta["translations"] = theta["translations"] + np.array([[10.0, -5.0, 2.0]])
        theta["bead_coords"] = theta["bead_coords"] + np.array([1.0, 1.0, 1.0])
        state_sync.apply(theta, self.layout, self.built)

        member_idx = self.layout.rigid_bodies[0].member_particle_indexes[0]
        member_particle = self.built.model.get_particle(IMP.ParticleIndex(int(member_idx)))
        moved_member_pos = np.array(IMP.core.XYZ(member_particle).get_coordinates())
        expected_member_pos = (
            self.layout.rigid_bodies[0].local_coordinates[0] + theta["translations"][0]
        )
        np.testing.assert_allclose(moved_member_pos, expected_member_pos, atol=1e-6)

        bead_pos = np.array(IMP.core.XYZ(self.layout.bead_particles[0]).get_coordinates())
        np.testing.assert_allclose(bead_pos, theta["bead_coords"][0], atol=1e-6)

    def test_template_capture_is_not_aliased(self):
        template_xyz, r = state_sync.capture_template(self.ji)
        original = template_xyz.copy()
        theta = state_sync.extract(self.built, self.layout)
        theta["bead_coords"] = theta["bead_coords"] + 500.0
        state_sync.apply(theta, self.layout, self.built)
        np.testing.assert_allclose(template_xyz, original)

    def test_expansion_matches_imp_score_after_random_moves(self):
        template_xyz, r = state_sync.capture_template(self.ji)
        expand = state_sync.make_expansion_fn(self.layout, template_xyz)

        rng = np.random.default_rng(1)
        theta = state_sync.extract(self.built, self.layout)
        theta["translations"] = theta["translations"] + rng.normal(scale=5.0, size=theta["translations"].shape)
        theta["bead_coords"] = theta["bead_coords"] + rng.normal(scale=5.0, size=theta["bead_coords"].shape)
        # small rotation perturbation, renormalized
        theta["quaternions"] = theta["quaternions"] + rng.normal(scale=0.05, size=theta["quaternions"].shape)
        theta["quaternions"] /= np.linalg.norm(theta["quaternions"], axis=-1, keepdims=True)

        xyz = np.array(expand(theta))
        jax_score = float(self.ji.score_func({"xyz": xyz, "r": r}))

        state_sync.apply(theta, self.layout, self.built)
        imp_score = self.sf.evaluate(False)

        self.assertAlmostEqual(jax_score, imp_score, places=3)

    def test_auxiliary_rows_do_not_affect_score(self):
        """Assumption 3 in doc/design.tex: rows outside the sampled leaves are
        frozen and, for the restraints exercised here, irrelevant to the score."""
        template_xyz, r = state_sync.capture_template(self.ji)
        expand = state_sync.make_expansion_fn(self.layout, template_xyz)
        theta = state_sync.extract(self.built, self.layout)

        baseline = float(self.ji.score_func({"xyz": np.array(expand(theta)), "r": r}))

        sampled_rows = set(self.layout.bead_particle_indexes.tolist())
        for rb in self.layout.rigid_bodies:
            sampled_rows.update(rb.member_particle_indexes.tolist())
        aux_rows = [i for i in range(template_xyz.shape[0]) if i not in sampled_rows]

        perturbed_template = template_xyz.copy()
        rng = np.random.default_rng(2)
        perturbed_template[aux_rows] += rng.normal(scale=50.0, size=(len(aux_rows), 3))
        expand_perturbed = state_sync.make_expansion_fn(self.layout, perturbed_template)
        perturbed_score = float(self.ji.score_func({"xyz": np.array(expand_perturbed(theta)), "r": r}))

        self.assertAlmostEqual(baseline, perturbed_score, places=6)


if __name__ == "__main__":
    unittest.main()
