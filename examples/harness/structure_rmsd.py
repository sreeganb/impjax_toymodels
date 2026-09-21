"""RMSD of a sampled model against the ground truth: the unshuffled build.

A toy model is built from a known structure, shuffled, and sampled back. The
ground truth is therefore not an external PDB at some other resolution -- it
is *the system itself, as built, before the shuffle*: the same coarse-grained
representation, the same beads, the same copy number, every rigid body sitting
on its input coordinates. That is `system.build_system(..., shuffle=False)`,
and it is the only reference this harness uses.

One number per model
--------------------
The RMSD is taken over every rigid-body bead in the whole system at once,
after a single optimal superposition (Kabsch, reflection-safe). Superposing is
required because nothing in the scoring function fixes the global frame --
every restraint is a function of internal distances -- so an un-superposed
RMSD would measure that irrelevant freedom instead of the structure.

Flexible beads are left out: PMI builds them on a placeholder position with no
structure behind it, so the unshuffled build has no "right answer" for them.

Identical copies are interchangeable. With copy_number > 1, which copy of a
protein ended up where the reference's copy 0 sits is arbitrary, so the RMSD
is minimised over relabellings of the copies (all proteins relabelled
together, since restraints tie copy i of one protein to copy i of another).
That is n! superpositions -- trivial for the handful of copies a toy model has,
and skipped above MAX_PERMUTED_COPIES.
"""

import itertools
from dataclasses import dataclass

import numpy as np

import IMP
import IMP.atom
import IMP.core

#: Above this many copies the n! relabellings are no longer tried; the model
#: is compared copy-for-copy in build order instead.
MAX_PERMUTED_COPIES = 5


@dataclass
class Reference:
    """The ground-truth structure of one case.

    coordinates : (n_copies, n_beads_per_copy, 3) rigid-body bead positions.
    imp_score : the full IMP score of the ground truth, under the same
        restraints the samplers see -- the score a perfect sampler would find
        (or beat, if the restraints are noisy), drawn as the target line on the
        score-convergence plot.
    """

    coordinates: np.ndarray
    imp_score: float


def copy_coordinates(root_hier, proteins, copy_number: int) -> np.ndarray:
    """Rigid-body bead coordinates, shaped (copy, bead, xyz).

    Ordering is molecule-major (the system module's fixed PROTEINS order) then
    representation order within a molecule, which is what IMP.atom.Selection
    returns. The reference and every trajectory are built by the same code, so
    row i is the same bead in both.
    """
    per_copy = []
    for copy_index in range(copy_number):
        rows = []
        for protein in proteins:
            particles = IMP.atom.Selection(
                root_hier, molecule=protein, copy_index=copy_index,
                resolution=1).get_selected_particles()
            rows.extend(IMP.core.XYZ(p).get_coordinates() for p in particles
                        if IMP.core.RigidMember.get_is_setup(p))
        per_copy.append(np.asarray([list(v) for v in rows], dtype=float))
    return np.stack(per_copy)


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD of two (n, 3) point sets after optimal rigid superposition.

    The determinant check forces a proper rotation: an unchecked SVD can return
    a reflection, which would score a mirror image as a perfect match.
    """
    mobile = mobile - mobile.mean(axis=0)
    target = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(mobile.T @ target)
    parity = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, parity]) @ u.T
    difference = (rotation @ mobile.T).T - target
    return float(np.sqrt((difference ** 2).sum(axis=1).mean()))


def rmsd_to_reference(model: np.ndarray, reference: np.ndarray) -> float:
    """One RMSD for the whole model, minimised over copy relabellings.

    Both arguments are (n_copies, n_beads, 3) arrays from copy_coordinates.
    """
    n_copies = reference.shape[0]
    target = reference.reshape(-1, 3)
    orders = (itertools.permutations(range(n_copies))
              if n_copies <= MAX_PERMUTED_COPIES else [tuple(range(n_copies))])
    return min(kabsch_rmsd(model[list(order)].reshape(-1, 3), target)
               for order in orders)


def build_reference(system, copy_number: int, distance_csv=None) -> Reference:
    """Build the case's ground truth: the same system, left unshuffled.

    `distance_csv` must be the restraint file the samplers use, so that the
    reference's IMP score is comparable to theirs.
    """
    built, score_function, _ = system.build_system(
        copy_number=copy_number, distance_csv=distance_csv, shuffle=False)
    return Reference(
        coordinates=copy_coordinates(built.root_hier, system.PROTEINS, copy_number),
        imp_score=float(score_function.evaluate(False)),
    )
