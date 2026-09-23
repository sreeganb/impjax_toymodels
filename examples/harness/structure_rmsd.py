"""RMSD of a sampled model against the ground truth: the unshuffled build.

A toy model is built from a known structure, shuffled, and sampled back. The
ground truth is therefore not an external PDB at some other resolution -- it
is *the system itself, as built, before the shuffle*: the same coarse-grained
representation, the same beads, the same copy number, every rigid body sitting
on its input coordinates. That is `system.build_system(..., shuffle=False)`,
written out as a one-frame RMF3 so it can be compared against, and looked at,
like any other model.

The RMSD itself is IMP's, not ours
----------------------------------
The measurement is `IMP.pmi.analysis.Precision`:

    pr = Precision(model, resolution=1, selection_dictionary={"selection": [...]})
    pr.add_structures(...); pr.set_reference_structure(reference_rmf, 0)
    pr.get_rmsd_wrt_reference_structure_with_alignment("set0", "selection")

which is exactly the call PMI_analysis's `accuracy.py::AccuracyModels` makes,
so this harness and the lab's standard analysis report the same quantity. It
aligns each model onto the reference over the selection and then takes the
RMSD; alignment is required because every restraint here is a function of
internal geometry only, so the assembly is free to sit anywhere in space.
`IMP.pmi.analysis.Alignment` also permutes identical copies, which is what
makes the number meaningful when copy_number > 1: which copy landed where is
arbitrary.

(The hand-written Kabsch version this replaced agreed with it to 1.5e-6 A on
real trajectory frames -- the change is about using the lab's own primitive,
not about a correction.)

Resolution 1 selects every bead of the representation, flexible beads
included. For a system with unstructured regions those beads sit on a PMI
placeholder in the reference and have no meaningful target, so give
`selection` only the molecules whose positions the ground truth defines.
"""

import contextlib
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import IMP
import IMP.pmi.analysis
import IMP.pmi.output


@dataclass
class Reference:
    """The ground truth of one case.

    rmf_path : one-frame RMF3 of the unshuffled build.
    imp_score : its IMP score under the same restraints the samplers see --
        the score a perfect sampler would find, drawn as the target line on
        the score-convergence plot. Note this is *not* necessarily the lowest
        score available: if the representation cannot reproduce the input
        structure exactly (coarse beads clashing across a tight interface,
        say), models that score better than the ground truth exist, and the
        gap between this number and what samplers reach is the model's own
        error rather than a sampling failure.
    """

    rmf_path: str
    imp_score: float


def write_reference(system, copy_number: int, distance_csv, rmf_path: str) -> Reference:
    """Build the case's ground truth and write it as a one-frame RMF3.

    `distance_csv` must be the restraint file the samplers use, so that the
    reference's IMP score is comparable to theirs.
    """
    built, score_function, _ = system.build_system(
        copy_number=copy_number, distance_csv=distance_csv, shuffle=False)
    score = float(score_function.evaluate(False))
    output = IMP.pmi.output.Output()
    output.init_rmf(rmf_path, [built.root_hier])
    output.write_rmf(rmf_path)
    output.close_rmf(rmf_path)
    return Reference(rmf_path=rmf_path, imp_score=score)


class RmsdToReference:
    """Measures frames of an RMF3 against a Reference, via PMI's Precision.

    One instance per case; `rmsd_for` is called once per trajectory file with
    all of the frames that file needs measured, since Precision reads frames
    in a batch and a fresh object per frame would re-read the reference every
    time.
    """

    def __init__(self, reference: Reference, proteins: Sequence[str]) -> None:
        self.reference = reference
        # "selection" is the key PMI_analysis uses; the alignment and the
        # measurement are both made over it, giving one global RMSD.
        self.selection_dictionary = {"selection": list(proteins)}

    def rmsd_for(self, rmf_path: str, frames: Sequence[int]) -> np.ndarray:
        """RMSD of the given frames of one RMF3 file, in the order asked for."""
        frames = list(frames)
        if not frames:
            return np.empty(0)
        model = IMP.Model()
        precision = IMP.pmi.analysis.Precision(
            model, resolution=1, selection_dictionary=self.selection_dictionary)
        precision.set_precision_style("pairwise_rmsd")
        # Precision narrates every frame it reads; that is one line per frame.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            precision.add_structures(zip([rmf_path] * len(frames), frames), "set0")
            precision.set_reference_structure(self.reference.rmf_path, 0)
            values = precision.get_rmsd_wrt_reference_structure_with_alignment(
                "set0", "selection")
        return np.asarray(values["selection"]["all_distances"], dtype=float)
