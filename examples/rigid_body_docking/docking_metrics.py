"""CAPRI-style accuracy metrics for the 1AVX docking benchmark.

`harness/evaluate_recovery.py` reports the global superposed RMSD over every
structured bead, which is the number that keeps this example comparable with
the KCOIL/ECOIL report.  It is not, on its own, the right question for a
docking problem: a global superposition distributes the error across both
partners, so a model whose two bodies are correct individually but wrongly
oriented relative to one another reports a middling number that says little
about whether the *interface* was found.

The docking field answers that with three separate measurements, and this
module computes all three, plus the CAPRI class they imply:

    L-RMSD   superpose the receptor, then measure the ligand.  All the error
             is forced onto the relative placement, which is the only thing
             sampling can get wrong here -- both bodies are rigid.
    I-RMSD   superpose and measure the interface beads alone.  Insensitive to
             a lever-arm effect that makes L-RMSD large when a distant part
             of the ligand swings, even though the contact is nearly right.
    fnat     fraction of the native inter-chain contacts the model recovers.
             Not a distance at all, which is the point: it degrades
             gracefully and stays interpretable when the RMSDs are large.

Bead resolution, and what that does to the thresholds
-----------------------------------------------------
CAPRI defines contacts between heavy atoms at 5 A and interface residues at
10 A.  This representation has no atoms: a resolution-2 bead has a radius of
2.9-4.2 A (mean 3.6).  A bead-centre distance therefore runs about two mean
radii longer than the closest heavy-atom approach, so the 5 A atomic
criterion corresponds to roughly 12 A here, which is the default below.  At
that cutoff the crystal structure has 74 native bead contacts across 47
interface beads -- fine enough for fnat to resolve about 1.4% per contact.

The CAPRI class is reported as an *indication*, not a certification: the
thresholds were calibrated on all-atom RMSDs, and a coarse-grained model
cannot be assessed to a 1 A "high quality" criterion in any strict sense.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

import IMP.atom
import IMP.core

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

import evaluate_recovery  # noqa: E402

#: Bead-centre stand-in for CAPRI's 5 A heavy-atom contact criterion; see the
#: module docstring for how it is derived from this representation's bead radii.
CONTACT_CUTOFF = 12.0

#: Which beads count as "the interface" for I-RMSD: those within this distance
#: of any bead of the other molecule, in the reference structure.
INTERFACE_CUTOFF = 12.0


@dataclass(frozen=True)
class DockingScores:
    """One model's accuracy against the reference, all four numbers together."""

    rmsd: float           # global, superposed over every structured bead
    ligand_rmsd: float    # receptor superposed, ligand measured
    interface_rmsd: float # interface beads only
    fnat: float           # fraction of native contacts recovered
    capri: str            # "high" / "medium" / "acceptable" / "incorrect"

    def as_row(self) -> dict:
        return {"rmsd": self.rmsd, "ligand_rmsd": self.ligand_rmsd,
                "interface_rmsd": self.interface_rmsd, "fnat": self.fnat,
                "capri": self.capri}


def protein_row_slices(root_hier, copy_index: int,
                       proteins: Sequence[str]) -> Dict[str, slice]:
    """Which rows of a `bead_coordinates` array belong to which molecule.

    `evaluate_recovery.bead_coordinates` concatenates molecules in `PROTEINS`
    order, so the split is a contiguous slice per molecule.  Derived from the
    hierarchy rather than hard-coded, so it stays correct if the
    coarse-graining changes.
    """
    slices: Dict[str, slice] = {}
    start = 0
    for protein in proteins:
        particles = IMP.atom.Selection(
            root_hier, molecule=protein, copy_index=copy_index,
            resolution=1).get_selected_particles()
        count = sum(1 for p in particles if IMP.core.RigidMember.get_is_setup(p))
        slices[protein] = slice(start, start + count)
        start += count
    return slices


def ligand_rmsd(mobile: np.ndarray, reference: np.ndarray,
                receptor: slice, ligand: slice) -> float:
    """CAPRI ligand RMSD: superpose on the receptor, measure the ligand.

    The transform is fitted on the receptor rows only and then applied to the
    ligand rows, which is why `kabsch_transform` returns the transform instead
    of an aligned copy.
    """
    transform = evaluate_recovery.kabsch_transform(mobile[receptor], reference[receptor])
    return evaluate_recovery.rmsd(
        evaluate_recovery.apply_transform(mobile[ligand], transform),
        reference[ligand])


def interface_beads(reference: np.ndarray, first: slice, second: slice,
                    cutoff: float = INTERFACE_CUTOFF) -> np.ndarray:
    """Row indexes of the beads forming the interface in the reference.

    Defined on the reference, never on the model: the question is whether the
    model reproduces *this* interface, so the bead set has to be fixed by the
    ground truth or a badly docked model would be scored against whatever
    accidental contact it happened to make.
    """
    separation = np.linalg.norm(
        reference[first][:, None, :] - reference[second][None, :, :], axis=-1)
    close = separation < cutoff
    return np.concatenate([
        np.where(close.any(axis=1))[0] + first.start,
        np.where(close.any(axis=0))[0] + second.start,
    ])


def interface_rmsd(mobile: np.ndarray, reference: np.ndarray,
                   beads: np.ndarray) -> float:
    """RMSD over the interface beads alone, after superposing on them."""
    if len(beads) == 0:
        return float("nan")
    return evaluate_recovery.superposed_rmsd(mobile[beads], reference[beads])


def native_contacts(reference: np.ndarray, first: slice, second: slice,
                    cutoff: float = CONTACT_CUTOFF) -> np.ndarray:
    """Boolean (n_first, n_second) mask of the reference's inter-chain contacts."""
    separation = np.linalg.norm(
        reference[first][:, None, :] - reference[second][None, :, :], axis=-1)
    return separation < cutoff


def fnat(mobile: np.ndarray, contacts: np.ndarray, first: slice, second: slice,
         cutoff: float = CONTACT_CUTOFF) -> float:
    """Fraction of the reference's inter-chain contacts present in the model.

    Translation- and rotation-invariant by construction -- it only ever looks
    at distances between the two molecules -- so unlike the RMSDs it needs no
    superposition at all.
    """
    total = int(contacts.sum())
    if total == 0:
        return float("nan")
    separation = np.linalg.norm(
        mobile[first][:, None, :] - mobile[second][None, :, :], axis=-1)
    return float((contacts & (separation < cutoff)).sum() / total)


def capri_class(fraction_native: float, l_rmsd: float, i_rmsd: float) -> str:
    """The CAPRI quality band implied by (fnat, L-RMSD, I-RMSD).

    Standard criteria.  Reported as an indication only: they were calibrated
    on all-atom RMSDs and this representation is coarse-grained (see the
    module docstring).
    """
    if fraction_native >= 0.5 and (l_rmsd <= 1.0 or i_rmsd <= 1.0):
        return "high"
    if fraction_native >= 0.3 and (l_rmsd <= 5.0 or i_rmsd <= 2.0):
        return "medium"
    if fraction_native >= 0.1 and (l_rmsd <= 10.0 or i_rmsd <= 4.0):
        return "acceptable"
    return "incorrect"


class DockingEvaluator:
    """Scores models against one reference, with the fixed sets precomputed.

    The row slices, interface bead set and native contact mask all depend only
    on the reference, so they are derived once here rather than per frame --
    a trajectory is thousands of frames and the contact mask is an
    (n_receptor x n_ligand) distance matrix.
    """

    def __init__(self, reference: np.ndarray, row_slices: Dict[str, slice],
                 proteins: Sequence[str],
                 contact_cutoff: float = CONTACT_CUTOFF,
                 interface_cutoff: float = INTERFACE_CUTOFF):
        # The larger molecule is the receptor, per the docking convention;
        # PROTEINS already lists it first for this system.
        self.receptor_name, self.ligand_name = proteins[0], proteins[1]
        self.receptor = row_slices[self.receptor_name]
        self.ligand = row_slices[self.ligand_name]
        self.reference = reference
        self.contact_cutoff = contact_cutoff
        self.contacts = native_contacts(
            reference, self.receptor, self.ligand, contact_cutoff)
        self.interface = interface_beads(
            reference, self.receptor, self.ligand, interface_cutoff)

    def score(self, mobile: np.ndarray) -> DockingScores:
        """Every metric for one model, as a single record."""
        fraction = fnat(mobile, self.contacts, self.receptor, self.ligand,
                        self.contact_cutoff)
        l_rmsd = ligand_rmsd(mobile, self.reference, self.receptor, self.ligand)
        i_rmsd = interface_rmsd(mobile, self.reference, self.interface)
        return DockingScores(
            rmsd=evaluate_recovery.superposed_rmsd(mobile, self.reference),
            ligand_rmsd=l_rmsd,
            interface_rmsd=i_rmsd,
            fnat=fraction,
            capri=capri_class(fraction, l_rmsd, i_rmsd),
        )

    def describe(self) -> str:
        """One line naming what the reference actually defines, for logs."""
        return (f"receptor {self.receptor_name} "
                f"({self.receptor.stop - self.receptor.start} beads), "
                f"ligand {self.ligand_name} "
                f"({self.ligand.stop - self.ligand.start} beads), "
                f"{int(self.contacts.sum())} native contacts within "
                f"{self.contact_cutoff} A, {len(self.interface)} interface beads")


def row_slices_for(system, data_dir: str = None) -> Dict[str, slice]:
    """The bead-row partition for `system`, from one unshuffled build.

    A property of the representation, not of any particular configuration, so
    every copy and every frame shares it.
    """
    built, _, _ = system.build_system(
        copy_number=1, data_dir=data_dir or system.DATA_DIR,
        shuffle=False, distance_csv=False)
    return protein_row_slices(built.root_hier, 0, system.PROTEINS)


def evaluators_for(system, references: Dict[int, np.ndarray],
                   data_dir: str = None, **cutoffs) -> Dict[int, DockingEvaluator]:
    """One evaluator per copy, sharing a single derivation of the bead layout.

    This is the entry point the benchmark harness calls (via a config's
    `metrics_module`): each copy is scored against its own copy of the ground
    truth, so each needs its own evaluator, but only the reference
    coordinates differ between them.
    """
    slices = row_slices_for(system, data_dir)
    return {index: DockingEvaluator(coordinates, slices, system.PROTEINS, **cutoffs)
            for index, coordinates in references.items()}


def build_evaluator(system, data_dir: str = None,
                    **cutoffs) -> Tuple[DockingEvaluator, np.ndarray]:
    """An evaluator against `system`'s own unshuffled build, plus that reference.

    The convenience path for scoring a single-copy trajectory directly; the
    benchmark uses `evaluators_for` instead, since it already has per-copy
    reference coordinates measured from the ground-truth structure.
    """
    built, _, _ = system.build_system(
        copy_number=1, data_dir=data_dir or system.DATA_DIR,
        shuffle=False, distance_csv=False)
    reference = evaluate_recovery.bead_coordinates(
        built.root_hier, copy_index=0, system=system)
    slices = protein_row_slices(built.root_hier, 0, system.PROTEINS)
    return DockingEvaluator(reference, slices, system.PROTEINS, **cutoffs), reference
