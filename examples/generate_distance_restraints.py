"""Derive harmonic distance restraints from the ground-truth KCOIL/ECOIL complex.

The point of this toy study is to start from a random configuration and
recover a structure we already know, so the restraints that drive the sampler
have to be *measured off* that known structure rather than guessed by hand.
This script does exactly that and writes the result to
examples/data/distance_constraints.csv, which kcoil_ecoil_system.py then reads
back in (see impjax_toymodels.distance_restraints for the file format).

How the target distances are obtained
-------------------------------------
The system is built exactly as it is for sampling, but with `shuffle=False`,
so every rigid body still sits on its PDB coordinates -- the ground truth --
and then `place_flexible_beads_at_reference` moves the linker beads onto their
reference positions too (PMI stacks them on a placeholder, because no
structure is read in for the unstructured region).  The result is a built
system that *is* the reference state.

Distances are then measured between the actual particles the restraints will
act on, in that state.  Measuring on the built system rather than
recomputing centers from the PDB by hand matters: IMP forms a resolution-2
bead from the mean of its two residue centers, not from the mean of all their
atoms, and with unequal atom counts per residue the two recipes differ by up
to ~0.5 A.  Reading the particles directly makes the written distance exactly
the distance the restraint will see at the ground truth, so the reference
state sits at the exact minimum of every restraint derived here.

Which pairs become restraints
-----------------------------
Every bead pair closer than `--cutoff` in the reference structure, minus the
pairs that carry no information:

* pairs inside the *same rigid body* -- their distance cannot change, so a
  restraint on them is a constant added to the score;
* pairs closer than `--min-seq-sep` residues along the same chain -- these
  merely restate the connectivity restraint that already links consecutive
  beads.

What is left is a contact-map-derived restraint set, the standard way to build
synthetic data for a structure-recovery benchmark, and the coarse-grained
analogue of a (very complete) crosslinking-MS dataset.
"""

import argparse
import itertools
import os
import sys
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

import IMP
import IMP.atom
import IMP.core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kcoil_ecoil_system as system_builder
from impjax_toymodels.distance_restraints import (
    ANY_COPY,
    DistanceConstraint,
    write_distance_constraints,
)

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


class Bead:
    """One coarse particle, with everything needed to restrain it."""

    def __init__(self, protein: str, residues: Sequence[int], rigid_body: str, center):
        self.protein = protein
        #: All residues represented by this bead; the first is what the CSV
        #: names, since selecting any of them returns this same particle.
        self.residues = tuple(residues)
        #: Name of the enclosing rigid body, or "" for a flexible bead.
        self.rigid_body = rigid_body
        self.center = center

    @property
    def residue(self) -> int:
        return self.residues[0]

    def __repr__(self) -> str:
        kind = self.rigid_body or "flex"
        return f"{self.protein}:{self.residues[0]}-{self.residues[-1]}({kind})"


def collect_beads(root_hier, protein: str) -> List[Bead]:
    """Enumerate copy 0's beads for one molecule, at their current positions.

    The caller is responsible for having put the system in the reference state
    first; this just reads coordinates off the particles.

    Only copy 0 is walked: every copy has an identical representation, and the
    restraints are written once per assembly and expanded over copies at load
    time (or by --explicit-copies below).
    """
    particles = IMP.atom.Selection(
        root_hier, molecule=protein, copy_index=0, resolution=1
    ).get_selected_particles()

    beads = []
    for particle in particles:
        if IMP.atom.Fragment.get_is_setup(particle):
            residues = [int(r) for r in IMP.atom.Fragment(particle).get_residue_indexes()]
        else:
            residues = [IMP.atom.Residue(particle).get_index()]
        rigid_body = ""
        if IMP.core.RigidMember.get_is_setup(particle):
            rigid_body = IMP.core.RigidMember(particle).get_rigid_body().get_name()
        center = np.array(IMP.core.XYZ(particle).get_coordinates())
        beads.append(Bead(protein, residues, rigid_body, center))
    return beads


def select_pairs(beads: Sequence[Bead], cutoff: float, min_seq_sep: int
                 ) -> List[Tuple[Bead, Bead, float]]:
    """Ground-truth contacts that actually constrain the degrees of freedom."""
    selected = []
    for first, second in itertools.combinations(beads, 2):
        if first.rigid_body and first.rigid_body == second.rigid_body:
            continue  # frozen distance: carries no information
        if first.protein == second.protein:
            separation = min(abs(a - b)
                             for a in first.residues for b in second.residues)
            if separation < min_seq_sep:
                continue  # already covered by the connectivity restraint
        distance = float(np.linalg.norm(first.center - second.center))
        if distance <= cutoff:
            selected.append((first, second, distance))
    return sorted(selected, key=lambda item: item[2])


def report_coverage(pairs: Sequence[Tuple[Bead, Bead, float]]) -> None:
    """Print how many restraints tie each pair of rigid bodies together.

    A rigid body needs at least three non-degenerate distances to another body
    to be positioned and oriented relative to it, so a pair count below three
    is a warning that the cutoff is too tight for the structure to be
    recoverable, however many restraints there are in total.
    """
    counts: Counter = Counter()
    for first, second, _ in pairs:
        key = tuple(sorted((first.rigid_body or f"{first.protein}/linker",
                            second.rigid_body or f"{second.protein}/linker")))
        counts[key] += 1
    print("\nrestraints per body pair (>= 3 needed to fix a relative pose):")
    for (left, right), count in sorted(counts.items()):
        flag = "" if count >= 3 or "linker" in left + right else "   <-- UNDER-DETERMINED"
        print(f"  {left:22s} -- {right:22s} {count:5d}{flag}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cutoff", type=float, default=12.0,
                        help="max ground-truth bead-bead distance to restrain, in A "
                             "(default: %(default)s)")
    parser.add_argument("--min-seq-sep", type=int, default=3,
                        help="skip same-chain pairs closer than this many residues, "
                             "since connectivity already covers them (default: %(default)s)")
    parser.add_argument("--force-constant", type=float, default=1.0,
                        help="harmonic kappa written to every row, in kcal/mol/A^2 "
                             "(default: %(default)s)")
    parser.add_argument("--explicit-copies", type=int, default=None,
                        help="write integer copy indexes for this many copies instead "
                             "of the copy-agnostic '*' wildcard; use when different "
                             "copies need different restraints")
    parser.add_argument("--data-dir", default=EXAMPLES_DIR,
                        help="base directory holding data/ (default: examples/)")
    parser.add_argument("--output", default=None,
                        help="output CSV (default: <data-dir>/data/distance_constraints.csv)")
    args = parser.parse_args(argv)

    output = args.output or os.path.join(args.data_dir, system_builder.DEFAULT_DISTANCE_CSV)

    # shuffle=False keeps the rigid bodies on their PDB coordinates and
    # distance_csv=False stops the builder demanding the file we are about to
    # write.  One copy is enough: copies are identical by construction.
    built, _, _ = system_builder.build_kcoil_ecoil_system(
        copy_number=1, data_dir=args.data_dir, shuffle=False, distance_csv=False)
    moved = system_builder.place_flexible_beads_at_reference(built, args.data_dir)
    print(f"placed {moved} flexible beads at their reference positions")

    beads: List[Bead] = []
    for protein in system_builder.PROTEINS:
        beads.extend(collect_beads(built.root_hier, protein))

    print(f"{len(beads)} beads across {len(system_builder.PROTEINS)} proteins")
    pairs = select_pairs(beads, args.cutoff, args.min_seq_sep)
    print(f"{len(pairs)} ground-truth contacts within {args.cutoff} A "
          f"(min seq sep {args.min_seq_sep})")
    if not pairs:
        parser.error("no contacts selected -- increase --cutoff")
    report_coverage(pairs)

    copy_indexes = ([(i, i) for i in range(args.explicit_copies)]
                    if args.explicit_copies else [(ANY_COPY, ANY_COPY)])
    constraints = [
        DistanceConstraint(
            residue1=first.residue, protein1=first.protein, copy1=copy1,
            residue2=second.residue, protein2=second.protein, copy2=copy2,
            distance=distance, force_constant=args.force_constant,
        )
        for copy1, copy2 in copy_indexes
        for first, second, distance in pairs
    ]
    write_distance_constraints(output, constraints)
    print(f"\nwrote {len(constraints)} restraints to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
