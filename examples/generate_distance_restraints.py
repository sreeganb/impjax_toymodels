"""Derive a sparse, crosslink-like restraint set from the ground-truth complex.

The point of this toy study is to start from a random configuration and
recover a structure we already know, so the restraints that drive the sampler
have to be measured off that structure.  But they also have to *look like data*
-- a crosslinking-MS experiment hands you a handful of residue pairs, not a
contact map -- because a dense restraint set both trivialises the inference and
costs real time: every restraint is a separate node in the exported JAX graph,
so hundreds of them dominate compile time and slow every score evaluation.

This script therefore mimics the chemistry.  Candidate pairs are restricted to
residues a crosslinker actually reacts with (`--residue-types`, default lysine
plus serine), ranked by their distance in the reference structure, and only the
closest `--top-n` are kept.  The result is written to
data/distance_constraints.csv, which kcoil_ecoil_system.py reads back in (see
impjax_toymodels.distance_restraints for the format).

Why lysine *and* serine here
----------------------------
NHS-ester crosslinkers (DSS, BS3) target lysine primary amines, and KCOIL and
ECOIL are rich in them -- but every one of those lysines is in a structured
domain.  The GGSGGGSGGG linker between the domains contains no lysine at all,
so a strictly lysine-only dataset says nothing whatsoever about the flexible
region.  NHS esters also have well-documented side-reactivity with the
hydroxyls of serine, threonine and tyrosine, and each linker carries serines at
residues 24 and 28, so including S puts a few genuine restraints on the
flexible beads instead of leaving them entirely data-free.  Pass
`--residue-types K` for the strict lysine-only set.

Where the coordinates come from
-------------------------------
The system is built exactly as it is for sampling but with `shuffle=False`, so
the rigid bodies still sit on their PDB coordinates -- the ground truth.  Their
bead centers are read straight off the built particles, because IMP forms a
resolution-2 bead from the mean of its two residue centers rather than the mean
of all their atoms, and with unequal atom counts the two recipes differ by up
to ~0.5 A here.

The unstructured linker has no structure read into the representation, so PMI
stacks its beads on a placeholder coordinate; those beads are deliberately left
where they are, and their reference positions are read from the PDB instead.
At resolution 1 a linker bead is a single residue, so its center is exactly
that residue's atom centroid and the two recipes coincide -- no discrepancy to
worry about, and nothing in the built system gets moved.

What is excluded
----------------
Pairs inside one rigid body (their distance cannot change, so the restraint is
a constant added to the score) and same-chain pairs closer than
`--min-seq-sep` residues (already covered by the connectivity restraint).
Several residues can share one coarse bead, so pairs that collapse onto the
same bead pair are deduplicated, keeping the closest.
"""

import argparse
import itertools
import os
import sys
from typing import Dict, List, Sequence, Tuple

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

#: Default crosslinker chemistry: lysine (the real NHS-ester target) plus
#: serine (documented side-reactivity), the latter being the only reactive
#: residue anywhere in the flexible linker.
DEFAULT_RESIDUE_TYPES = "KS"


class Bead:
    """One coarse particle, with everything needed to restrain it."""

    def __init__(self, protein: str, residues: Sequence[int], rigid_body: str, center):
        self.protein = protein
        #: All residues represented by this bead; the CSV names whichever one
        #: carried the reactive side chain, since selecting any of them
        #: returns this same particle.
        self.residues = tuple(residues)
        #: Name of the enclosing rigid body, or "" for a flexible bead.
        self.rigid_body = rigid_body
        self.center = center

    @property
    def body(self) -> str:
        """Label used in the coverage report."""
        return self.rigid_body or f"{self.protein}/linker"

    def __repr__(self) -> str:
        return f"{self.protein}:{self.residues[0]}-{self.residues[-1]}({self.body})"


def read_sequence(protein: str, data_dir: str) -> str:
    """One-letter sequence of `protein`, from the same FASTA the build uses."""
    info = system_builder._load(
        os.path.join(data_dir, "data", "json_files", f"{protein}.json"))
    letters = []
    with open(os.path.join(data_dir, info["files"]["fasta"])) as handle:
        for line in handle:
            if not line.startswith(">"):
                letters.append(line.strip())
    return "".join(letters)


def reference_linker_centers(protein: str, data_dir: str) -> Dict[int, np.ndarray]:
    """Per-residue centroid from the reference PDB, for the flexible beads.

    Only consulted for single-residue beads, where the centroid of the
    residue's atoms is exactly the center IMP would place.
    """
    info = system_builder._load(
        os.path.join(data_dir, "data", "json_files", f"{protein}.json"))
    chain = info["monomer_chain"][0]
    atoms: Dict[int, List[List[float]]] = {}
    with open(os.path.join(data_dir, info["files"]["pdb"])) as handle:
        for line in handle:
            if line.startswith("ATOM") and line[21] == chain:
                atoms.setdefault(int(line[22:26]), []).append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {residue: np.mean(coords, axis=0) for residue, coords in atoms.items()}


def collect_beads(root_hier, protein: str, data_dir: str) -> Dict[int, Bead]:
    """Map every residue of copy 0 to its bead, at ground-truth coordinates.

    Only copy 0 is walked: every copy has an identical representation, and the
    restraints are written once per assembly and expanded over copies at load
    time (or by --explicit-copies).
    """
    particles = IMP.atom.Selection(
        root_hier, molecule=protein, copy_index=0, resolution=1).get_selected_particles()
    linker_centers = reference_linker_centers(protein, data_dir)

    by_residue: Dict[int, Bead] = {}
    for particle in particles:
        if IMP.atom.Fragment.get_is_setup(particle):
            residues = [int(r) for r in IMP.atom.Fragment(particle).get_residue_indexes()]
        else:
            residues = [IMP.atom.Residue(particle).get_index()]

        if IMP.core.RigidMember.get_is_setup(particle):
            rigid_body = IMP.core.RigidMember(particle).get_rigid_body().get_name()
            center = np.array(IMP.core.XYZ(particle).get_coordinates())
        else:
            # Flexible bead: left untouched in the model, so take its reference
            # position from the PDB rather than from the placeholder it sits on.
            rigid_body = ""
            if len(residues) != 1:
                raise ValueError(
                    f"{protein}: flexible bead spans {residues}; the reference "
                    "position is only exact for single-residue beads")
            if residues[0] not in linker_centers:
                raise ValueError(
                    f"{protein}: reference PDB has no residue {residues[0]}")
            center = linker_centers[residues[0]]

        bead = Bead(protein, residues, rigid_body, center)
        for residue in residues:
            by_residue[residue] = bead
    return by_residue


def candidate_pairs(beads1: Dict[int, Bead], residues1: Sequence[int],
                    beads2: Dict[int, Bead], residues2: Sequence[int],
                    min_seq_sep: int, same_molecule: bool
                    ) -> List[Tuple[float, int, Bead, int, Bead]]:
    """Reactive-residue pairs, distance-ranked and deduplicated by bead pair."""
    best: Dict[Tuple[int, int], Tuple[float, int, Bead, int, Bead]] = {}
    combinations = (itertools.combinations(residues1, 2) if same_molecule
                    else itertools.product(residues1, residues2))
    for residue1, residue2 in combinations:
        bead1 = beads1[residue1]
        bead2 = (beads1 if same_molecule else beads2)[residue2]
        if bead1 is bead2:
            continue
        if bead1.rigid_body and bead1.rigid_body == bead2.rigid_body:
            continue  # frozen distance: carries no information
        if same_molecule and min(abs(a - b) for a in bead1.residues
                                 for b in bead2.residues) < min_seq_sep:
            continue  # already covered by the connectivity restraint
        distance = float(np.linalg.norm(bead1.center - bead2.center))
        # Several reactive residues can share one bead; keep the closest, so a
        # single bead pair never contributes two near-identical restraints.
        key = (id(bead1), id(bead2))
        if key not in best or distance < best[key][0]:
            best[key] = (distance, residue1, bead1, residue2, bead2)
    return sorted(best.values(), key=lambda item: item[0])


def report_selection(selected, label: str) -> None:
    """Print what was chosen and which bodies it ties together."""
    print(f"\n{label}: {len(selected)} restraint(s)")
    for distance, residue1, bead1, residue2, bead2 in selected:
        flexible = "*" if not (bead1.rigid_body and bead2.rigid_body) else " "
        print(f"  {flexible} {bead1.protein} {residue1:>3d} -- "
              f"{bead2.protein} {residue2:>3d} : {distance:6.2f} A   "
              f"[{bead1.body} -- {bead2.body}]")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-n", type=int, default=10,
                        help="inter-molecular restraints per copy pair, taken as the "
                             "closest reactive-residue pairs (default: %(default)s)")
    parser.add_argument("--top-n-intra", type=int, default=2,
                        help="intra-molecular restraints per protein, tying its two "
                             "domains together (default: %(default)s)")
    parser.add_argument("--residue-types", default=DEFAULT_RESIDUE_TYPES,
                        help="one-letter codes the crosslinker reacts with "
                             "(default: %(default)s; use K for lysine only)")
    parser.add_argument("--min-seq-sep", type=int, default=3,
                        help="skip same-chain pairs closer than this many residues, "
                             "since connectivity already covers them (default: %(default)s)")
    parser.add_argument("--force-constant", type=float, default=1.0,
                        help="harmonic kappa written to every row, in kcal/mol/A^2 "
                             "(default: %(default)s)")
    parser.add_argument("--explicit-copies", type=int, default=None,
                        help="write integer copy indexes for this many copies instead "
                             "of the copy-agnostic '*' wildcard")
    parser.add_argument("--data-dir", default=EXAMPLES_DIR,
                        help="base directory holding data/ (default: examples/)")
    parser.add_argument("--output", default=None,
                        help="output CSV (default: <data-dir>/data/distance_constraints.csv)")
    args = parser.parse_args(argv)

    output = args.output or os.path.join(args.data_dir, system_builder.DEFAULT_DISTANCE_CSV)
    reactive_types = set(args.residue_types.upper())

    # shuffle=False keeps the rigid bodies on their PDB coordinates and
    # distance_csv=False stops the builder demanding the file we are about to
    # write.  One copy is enough: copies are identical by construction.
    built, _, _ = system_builder.build_kcoil_ecoil_system(
        copy_number=1, data_dir=args.data_dir, shuffle=False, distance_csv=False)

    beads, reactive = {}, {}
    for protein in system_builder.PROTEINS:
        beads[protein] = collect_beads(built.root_hier, protein, args.data_dir)
        sequence = read_sequence(protein, args.data_dir)
        reactive[protein] = [index for index, letter in enumerate(sequence, start=1)
                             if letter in reactive_types]
        in_linker = [r for r in reactive[protein] if not beads[protein][r].rigid_body]
        print(f"{protein}: {len(reactive[protein])} reactive residue(s) "
              f"[{args.residue_types}], {len(in_linker)} of them in the flexible linker")

    first, second = system_builder.PROTEINS
    inter = candidate_pairs(beads[first], reactive[first],
                            beads[second], reactive[second],
                            args.min_seq_sep, same_molecule=False)[:args.top_n]
    report_selection(inter, f"{first} <-> {second} (* = involves a flexible bead)")

    selected = list(inter)
    for protein in system_builder.PROTEINS:
        intra = candidate_pairs(beads[protein], reactive[protein],
                                beads[protein], reactive[protein],
                                args.min_seq_sep, same_molecule=True)[:args.top_n_intra]
        report_selection(intra, f"{protein} internal")
        selected.extend(intra)

    if not selected:
        parser.error("no restraints selected -- widen --residue-types or raise --top-n")

    copy_indexes = ([(i, i) for i in range(args.explicit_copies)]
                    if args.explicit_copies else [(ANY_COPY, ANY_COPY)])
    constraints = [
        DistanceConstraint(
            residue1=residue1, protein1=bead1.protein, copy1=copy1,
            residue2=residue2, protein2=bead2.protein, copy2=copy2,
            distance=distance, force_constant=args.force_constant,
        )
        for copy1, copy2 in copy_indexes
        for distance, residue1, bead1, residue2, bead2 in selected
    ]
    write_distance_constraints(output, constraints)
    print(f"\nwrote {len(constraints)} restraint(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
