"""Score a sampled trajectory against the ground-truth KCOIL/ECOIL structure.

The restraints in data/distance_constraints.csv were measured off a known
reference complex, so the question a run has to answer is not "did the score
go down" but "did the model come back to that structure".  This script answers
it directly: it walks an RMF3 trajectory and reports, per frame, the RMSD to
the reference after optimal superposition.

Superposition is the right comparison here because every restraint in play --
distance, connectivity, excluded volume -- is a function of internal geometry
only.  Nothing in the scoring function knows where the origin is or which way
is up, so the recovered assembly is free to sit anywhere in space, and an
un-superposed RMSD would measure that irrelevant freedom instead of the
structure.

Copies are scored one at a time, each against the single reference dimer,
and the reported number is the worst copy in the frame.  That is deliberate:
the restraint file ties copy i of KCOIL to copy i of ECOIL and says nothing
across copies, so a multi-copy model is N independent dimers that are free to
land anywhere relative to each other.  Superposing all of them at once would
measure that irrelevant freedom; scoring the *worst* copy asks the question
that matters, which is whether every dimer came back.

Usage
-----
    python evaluate_recovery.py out/kcoil_ecoil_smc_adaptive.rmf3
    python evaluate_recovery.py out/*.rmf3 --copy-number 2
"""

import argparse
import os
import sys

import numpy as np

import IMP
import IMP.atom
import IMP.core
import IMP.rmf
import RMF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kcoil_ecoil_system as system_builder

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


def bead_coordinates(root_hier, copy_index: int) -> np.ndarray:
    """One copy's beads (both proteins), in a fixed, reproducible order.

    Molecule-major then representation order, which is what
    `IMP.atom.Selection` yields for a given (molecule, copy) pair.  Both the
    reference and the trajectory system are read with this same function, and
    both are built by the same code, so row i means the same bead in both.
    """
    coordinates = []
    for protein in system_builder.PROTEINS:
        particles = IMP.atom.Selection(
            root_hier, molecule=protein, copy_index=copy_index,
            resolution=1).get_selected_particles()
        coordinates.extend(
            list(IMP.core.XYZ(particle).get_coordinates()) for particle in particles)
    return np.asarray(coordinates)


def superposed_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    """RMSD after the optimal rigid superposition (Kabsch, reflection-safe)."""
    mobile_centered = mobile - mobile.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(mobile_centered.T @ target_centered)
    # Force a proper rotation: an unchecked SVD can return a reflection, which
    # would report a mirror image of the structure as a perfect match.
    parity = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, parity]) @ u.T
    residual = target_centered - (rotation @ mobile_centered.T).T
    return float(np.sqrt((residual ** 2).sum(axis=1).mean()))


def reference_coordinates(data_dir: str) -> np.ndarray:
    """The ground truth dimer: an unshuffled build with its linkers put back.

    Built at copy_number=1 because there is only ever one reference dimer;
    every copy in a multi-copy model is compared against this same one.
    """
    built, _, _ = system_builder.build_kcoil_ecoil_system(
        copy_number=1, data_dir=data_dir, shuffle=False)
    system_builder.place_flexible_beads_at_reference(built, data_dir)
    return bead_coordinates(built.root_hier, copy_index=0)


def trajectory_rmsds(rmf_path: str, copy_number: int, data_dir: str,
                     reference: np.ndarray) -> np.ndarray:
    """RMSD of every frame in one RMF3 file against the reference."""
    # A fresh system is built and *linked* to the file, so the trajectory's
    # frames are loaded into particles laid out exactly like the reference's.
    built, _, _ = system_builder.build_kcoil_ecoil_system(
        copy_number=copy_number, data_dir=data_dir)
    handle = RMF.open_rmf_file_read_only(rmf_path)
    IMP.rmf.create_hierarchies(handle, built.model)
    IMP.rmf.link_hierarchies(handle, [built.root_hier])

    values = []
    for frame in range(handle.get_number_of_frames()):
        IMP.rmf.load_frame(handle, RMF.FrameID(frame))
        # Worst copy in the frame -- see the module docstring for why the
        # copies are scored separately rather than superposed together.
        values.append(max(
            superposed_rmsd(bead_coordinates(built.root_hier, copy_index), reference)
            for copy_index in range(copy_number)))
    return np.asarray(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rmf", nargs="+", help="RMF3 trajectory file(s) to score")
    parser.add_argument("--copy-number", type=int, default=1,
                        help="copy number the trajectory was sampled at (default: %(default)s)")
    parser.add_argument("--data-dir", default=EXAMPLES_DIR,
                        help="base directory holding data/ (default: examples/)")
    args = parser.parse_args(argv)

    reference = reference_coordinates(args.data_dir)
    print(f"reference dimer: {len(reference)} beads; scoring {args.copy_number} "
          f"cop{'y' if args.copy_number == 1 else 'ies'} per frame\n")
    print(f"{'trajectory':40s} {'frames':>7s} {'first':>8s} {'last':>8s} "
          f"{'best':>8s} {'@frame':>7s}")
    for path in args.rmf:
        values = trajectory_rmsds(path, args.copy_number, args.data_dir, reference)
        if not len(values):
            print(f"{os.path.basename(path):40s} {'0':>7s}  (empty)")
            continue
        print(f"{os.path.basename(path):40s} {len(values):7d} {values[0]:8.2f} "
              f"{values[-1]:8.2f} {values.min():8.2f} {int(values.argmin()):7d}")
    print("\nRMSD is in angstroms, over one copy's coarse beads, after optimal "
          "superposition;\nwith more than one copy, the worst copy in the frame "
          "is reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
