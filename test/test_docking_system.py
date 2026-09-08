"""Tests for the 1AVX rigid-body docking system and its synthetic restraints.

Three things have to hold for this example to mean anything:

* the built system really is two rigid bodies and nothing else, so it is a
  docking problem rather than a refinement;
* the ground truth really is the minimum of the restraints that were measured
  off it, otherwise "recovered the structure" and "lowered the score" are
  different questions and no report can be trusted;
* the prior/likelihood split is disjoint, or the posterior double-counts.

The fourth test class locks in a measured property of IMP's JAX export that
this system depends on -- see docking_system's module docstring.
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import IMP
import IMP.core
import IMP.pmi.restraints.basic

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
DOCKING_DIR = os.path.join(EXAMPLES_DIR, "rigid_body_docking")
for _directory in (DOCKING_DIR, os.path.join(EXAMPLES_DIR, "harness")):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import docking_system  # noqa: E402

from impjax_toymodels import contact_map, dof_layout, proposals, state_sync  # noqa: E402

DATA_DIR = os.path.join(DOCKING_DIR, "data")
CONTACT_MAP = os.path.join(DATA_DIR, "contact_maps", "1avx_n1.csv")
CONSTRAINTS = os.path.join(DATA_DIR, "distance_constraints.csv")

#: Beads per chain at resolution 2 (223 and 172 residues).
EXPECTED_MEMBERS = (112, 86)


class DockingRepresentationTests(unittest.TestCase):
    """The representation is exactly two rigid bodies, no flexible beads."""

    @classmethod
    def setUpClass(cls):
        cls.built, cls.score_function, cls.output_objects = (
            docking_system.build_system(shuffle=False))

    def test_two_rigid_bodies_and_no_flexible_beads(self):
        rigid_bodies, beads = self.built.rigid_bodies_and_beads()
        self.assertEqual(len(rigid_bodies), 2)
        self.assertEqual(len(beads), 0)

    def test_layout_has_only_rigid_body_degrees_of_freedom(self):
        layout = dof_layout.build(self.built)
        self.assertEqual(layout.n_rigid_bodies, 2)
        self.assertEqual(layout.n_beads, 0)
        # 4 (quaternion) + 3 (translation) per body; nothing else is sampled.
        self.assertEqual(layout.flat_size, 14)
        self.assertEqual(
            tuple(len(rb.member_particle_indexes) for rb in layout.rigid_bodies),
            EXPECTED_MEMBERS)

    def test_no_connectivity_restraint_when_there_are_no_beads(self):
        """A one-rigid-body chain's connectivity term is a constant, so it is
        omitted rather than scored -- every restraint costs a JAX graph node."""
        names = [r.get_name() for r in self.score_function.get_restraints()]
        self.assertFalse([n for n in names if "onnectivity" in n], names)
        self.assertIn("ExcludedVolumeSphere", names)

    def test_rigid_body_ordering_matches_the_protein_order(self):
        """PROTEINS fixes bead ordering, which is what lets a reference build
        and a trajectory build be compared row for row."""
        self.assertEqual(docking_system.PROTEINS, ("TRYP", "STI"))
        layout = dof_layout.build(self.built)
        # TRYP is the larger chain, so it must come first.
        self.assertGreater(len(layout.rigid_bodies[0].member_particle_indexes),
                           len(layout.rigid_bodies[1].member_particle_indexes))


class GroundTruthTests(unittest.TestCase):
    """The crystal pose must be the minimum of the restraints taken from it."""

    def test_distance_restraints_vanish_at_the_crystal_pose(self):
        _, _, output_objects = docking_system.build_system(shuffle=False)
        restraints = [o for o in output_objects
                      if isinstance(o, IMP.pmi.restraints.basic.DistanceRestraint)]
        self.assertEqual(len(restraints), 20)

        scores = np.array([r.get_restraint().get_score() for r in restraints])
        # Each well is 0.5*kappa*(d-d0)^2 with kappa = 1, so the residual
        # deviation is sqrt(2*score). The CSV stores targets to 3 decimals,
        # which bounds how close this can get.
        self.assertLess(np.sqrt(2 * scores.max()), 1e-3)

    def test_shuffling_moves_the_system_far_off_the_answer(self):
        """Inference has to start away from the answer, and by a real margin."""
        _, score_function, output_objects = docking_system.build_system(shuffle=True)
        restraints = [o for o in output_objects
                      if isinstance(o, IMP.pmi.restraints.basic.DistanceRestraint)]
        scores = np.array([r.get_restraint().get_score() for r in restraints])
        self.assertGreater(np.sqrt(2 * scores.max()), 10.0)


class RestraintSelectionTests(unittest.TestCase):
    """The selected restraints are interface crosslinks, and only those."""

    @classmethod
    def setUpClass(cls):
        from impjax_toymodels import distance_restraints
        cls.constraints = distance_restraints.read_distance_constraints(CONSTRAINTS)
        cls.pairs = contact_map.read_contact_map(CONTACT_MAP)

    def test_every_restraint_spans_the_two_molecules(self):
        """With one rigid body per chain, an intra-chain pair is frozen and
        carries no information; contact_map.select drops those as `same_body`,
        so nothing should have survived except inter-molecular pairs."""
        for constraint in self.constraints:
            self.assertNotEqual(constraint.protein1, constraint.protein2)

    def test_restraints_are_crosslinker_reactive_residues(self):
        """Selected by chemistry, not merely by proximity: NHS esters react
        with lysine and with serine/threonine/tyrosine hydroxyls."""
        reactive = {"K", "S"}
        selected = {(p.protein1, p.residue1, p.restype1) for p in self.pairs}
        by_residue = {(protein, residue): restype
                      for protein, residue, restype in selected}
        for constraint in self.constraints:
            restype = by_residue.get((constraint.protein1, constraint.residue1))
            if restype is not None:
                self.assertIn(restype, reactive)

    def test_contact_map_covers_both_chains(self):
        kinds = {pair.kind for pair in self.pairs}
        self.assertIn(contact_map.INTER_MOL, kinds)
        proteins = {pair.protein1 for pair in self.pairs} | {
            pair.protein2 for pair in self.pairs}
        self.assertEqual(proteins, set(docking_system.PROTEINS))


class SplitTests(unittest.TestCase):
    """The prior/likelihood partition must be disjoint and correctly assigned."""

    @classmethod
    def setUpClass(cls):
        (cls.built, cls.likelihood_sf, cls.prior_sf,
         cls.output_objects) = docking_system.build_split(shuffle=False)

    def _names(self, score_function):
        return {r.get_name() for r in score_function.get_restraints()}

    def test_split_scoring_functions_are_disjoint(self):
        # priors.restraint_prior raises on an overlap, because a restraint in
        # both would be counted twice at lambda = 1.
        self.assertFalse(self._names(self.likelihood_sf)
                         & self._names(self.prior_sf))

    def test_likelihood_holds_the_data_and_prior_holds_excluded_volume(self):
        """The data-derived restraints are tempered; excluded volume is
        structural prior knowledge and is not."""
        self.assertEqual(len(self.likelihood_sf.get_restraints()), 20)
        self.assertIn("ExcludedVolumeSphere", self._names(self.prior_sf))
        self.assertNotIn("ExcludedVolumeSphere", self._names(self.likelihood_sf))

    def test_likelihood_varies_across_configurations(self):
        """The trap the KCOIL/ECOIL split fell into: a likelihood with no
        variance across prior draws makes tempering a no-op. Interface
        restraints have plenty."""
        at_truth = float(self.likelihood_sf.evaluate(False))
        _, shuffled_likelihood, _, _ = docking_system.build_split(shuffle=True)
        shuffled = float(shuffled_likelihood.evaluate(False))
        self.assertLess(at_truth, 1e-3)
        self.assertGreater(shuffled, 100.0)


class ExcludedVolumeOffsetTests(unittest.TestCase):
    """IMP's all-pairs JAX excluded volume differs from its CPU score by a
    constant -- locked in here because the correctness of sampling rests on
    that difference being exactly constant, not merely small.

    See docking_system's module docstring for the measured decomposition.
    """

    def test_intra_body_contribution_is_invariant_under_rigid_motion(self):
        built, score_function, _ = docking_system.build_system(
            shuffle=False, distance_csv=False)
        score_function.evaluate(False)  # materialize the JAX export

        layout = dof_layout.build(built)
        jax_interface = score_function._get_jax()
        template, radii = state_sync.capture_template(jax_interface)
        expand = state_sync.make_expansion_fn(layout, template)

        groups = [rb.member_particle_indexes for rb in layout.rigid_bodies]
        indexes = np.concatenate(groups)
        body = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
        upper = np.triu_indices(len(indexes), 1)
        same_body = body[upper[0]] == body[upper[1]]
        member_radii = radii[indexes]

        def parts(theta):
            """(intra-body, inter-body) halves of the all-pairs soft-sphere sum."""
            points = np.asarray(expand(theta))[indexes]
            separation = np.linalg.norm(
                points[:, None, :] - points[None, :, :], axis=-1)
            overlap = np.maximum(
                member_radii[:, None] + member_radii[None, :] - separation, 0.0)
            return (0.5 * (overlap[upper][same_body] ** 2).sum(),
                    0.5 * (overlap[upper][~same_body] ** 2).sum())

        proposal = proposals.build_composite(
            layout, sigma_rotation=0.3, sigma_translation=20.0,
            sigma_bead=1.0, mode="rigid")
        theta = {k: jnp.asarray(v)
                 for k, v in state_sync.extract(built, layout).items()}

        intra_values, inter_values = [], []
        for step in range(5):
            intra, inter = parts(theta)
            intra_values.append(intra)
            inter_values.append(inter)
            theta = proposal(jax.random.PRNGKey(step), theta)

        # Constant to float32 rounding in the quaternion rotation...
        self.assertLess(max(intra_values) - min(intra_values), 1e-2)
        # ...while the inter-body term genuinely moves, so this is not just a
        # test that nothing happened.
        self.assertGreater(max(inter_values) - min(inter_values), 1.0)


if __name__ == "__main__":
    unittest.main()
