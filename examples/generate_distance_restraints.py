"""Select a sparse restraint set out of a ground-truth contact map.

This is the cheap, run-time half of the restraint pipeline.  The expensive
half -- inventorying every residue pair in a known structure -- happens once,
in generate_contact_map.py.  Here we only rank, filter and take the top N, so
sweeping "how many restraints?" or "which crosslinker chemistry?" costs a file
read rather than any recomputation:

    generate_contact_map.py        ground truth PDB  ->  contact map   (once)
    generate_distance_restraints.py  contact map     ->  restraint CSV (per run)
    kcoil_ecoil_system.py            restraint CSV   ->  IMP restraints

Sparse is the point, and not only for realism: every restraint becomes a
separate node in the exported JAX graph, so the count is a direct cost.  On
this system 14 restraints compile in 0.4 s and score in 0.088 ms, while 350
take 11.8 s and 0.692 ms.

Residue distances rank, bead distances restrain
-----------------------------------------------
The contact map stores residue-centroid distances, which is what makes it
representation-independent -- it describes the structure, not the
coarse-graining.  Those are used only to decide *which* pairs to keep.

The distance actually written into a restraint must be the distance between
the two beads the restraint will act on, or the ground truth will not sit at
the restraint's minimum: at resolution 2 a bead covers two residues and its
center is not either residue's centroid.  Those bead centers are computed here
from the ground-truth PDB as the plain mean of the constituent residues' atoms
-- verified to reproduce IMP's own bead placement to ~1e-07 A across every
bead in this system.  Computing them from the structure rather than reading
them off a built model is what lets a genuine multi-copy ground truth work,
where each copy sits at different coordinates.
"""

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

import IMP
import IMP.atom
import IMP.core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kcoil_ecoil_system as system_builder
from generate_contact_map import parse_chain_spec
from impjax_toymodels.contact_map import read_contact_map, select
from impjax_toymodels.distance_restraints import (
    ANY_COPY,
    DistanceConstraint,
    write_distance_constraints,
)

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

#: Lysine is the real NHS-ester (DSS/BS3) target, but every lysine in this
#: system sits in a structured domain -- the GGSGGGSGGG linker has none, so a
#: lysine-only dataset says nothing about the flexible region.  NHS esters also
#: react with serine/threonine/tyrosine hydroxyls, and each linker carries
#: serines at 24 and 28.
DEFAULT_RESIDUE_TYPES = "KS"


def residue_to_bead(root_hier, protein: str) -> Dict[int, Tuple[int, ...]]:
    """Map each residue to the tuple of residues sharing its bead.

    Read from copy 0: every copy has an identical representation, so the
    decomposition is a property of the protein, not of the copy.
    """
    mapping: Dict[int, Tuple[int, ...]] = {}
    particles = IMP.atom.Selection(
        root_hier, molecule=protein, copy_index=0, resolution=1).get_selected_particles()
    for particle in particles:
        if IMP.atom.Fragment.get_is_setup(particle):
            residues = tuple(int(r) for r in
                             IMP.atom.Fragment(particle).get_residue_indexes())
        else:
            residues = (IMP.atom.Residue(particle).get_index(),)
        for residue in residues:
            mapping[residue] = residues
    return mapping


def chain_atoms(pdb_path: str, chains: Dict[str, Tuple[str, int]]
                ) -> Dict[Tuple[str, int], Dict[int, np.ndarray]]:
    """Per-residue atom coordinates of the ground truth, keyed by (protein, copy)."""
    atoms: Dict[Tuple[str, int], Dict[int, list]] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain not in chains:
                continue
            key = chains[chain]
            atoms.setdefault(key, {}).setdefault(int(line[22:26]), []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {key: {residue: np.asarray(coords) for residue, coords in residues.items()}
            for key, residues in atoms.items()}


def bead_center(atoms: Dict[int, np.ndarray], residues: Sequence[int]) -> np.ndarray:
    """Ground-truth center of the bead covering `residues`.

    The plain unweighted mean over every atom of every constituent residue --
    not the mean of the residues' own centroids, which is a different number
    whenever the residues have unequal atom counts (out by up to 0.5 A here).
    """
    stacked = np.concatenate([atoms[residue] for residue in residues], axis=0)
    return stacked.mean(axis=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contact-map", required=True,
                        help="contact-map CSV from generate_contact_map.py")
    parser.add_argument("--structure", required=True,
                        help="the same ground-truth PDB the map was built from")
    parser.add_argument("--chains", nargs="+", required=True, metavar="CHAIN=PROTEIN:COPY",
                        help="chain mapping, matching the one used for the map")
    parser.add_argument("--top-n", type=int, default=10,
                        help="restraints per assembly-spanning group (default: %(default)s)")
    parser.add_argument("--top-n-intra", type=int, default=2,
                        help="restraints per chain, tying its domains together "
                             "(default: %(default)s)")
    parser.add_argument("--residue-types", default=DEFAULT_RESIDUE_TYPES,
                        help="one-letter codes the crosslinker reacts with; pass ALL to "
                             "ignore residue identity (default: %(default)s)")
    parser.add_argument("--min-seq-sep", type=int, default=3,
                        help="skip same-chain pairs closer than this many residues "
                             "(default: %(default)s)")
    parser.add_argument("--force-constant", type=float, default=1.0,
                        help="harmonic kappa, kcal/mol/A^2 (default: %(default)s)")
    parser.add_argument("--wildcard-copies", action="store_true",
                        help="write '*' in the copy columns instead of the map's copy "
                             "indexes, so one restraint file replicates to any "
                             "--copy-number; only valid when no selected pair crosses "
                             "copies")
    parser.add_argument("--data-dir", default=EXAMPLES_DIR)
    parser.add_argument("--output", default=None,
                        help="output CSV (default: <data-dir>/data/distance_constraints.csv)")
    args = parser.parse_args(argv)

    output = args.output or os.path.join(args.data_dir, system_builder.DEFAULT_DISTANCE_CSV)
    residue_types = None if args.residue_types.upper() == "ALL" else set(
        args.residue_types.upper())

    chains = dict(
        (chain, (protein, copy_index))
        for chain, protein, copy_index in map(parse_chain_spec, args.chains))
    atoms = chain_atoms(args.structure, chains)

    # Built only for its bead decomposition -- no coordinates are read from it,
    # so its configuration is irrelevant.
    built, _, _ = system_builder.build_kcoil_ecoil_system(
        copy_number=1, data_dir=args.data_dir, distance_csv=False)
    beads = {protein: residue_to_bead(built.root_hier, protein)
             for protein in system_builder.PROTEINS}

    def bead_key(pair):
        """Identity of the bead pair a contact would restrain."""
        return (pair.protein1, pair.copy1, beads[pair.protein1][pair.residue1],
                pair.protein2, pair.copy2, beads[pair.protein2][pair.residue2])

    pairs = read_contact_map(args.contact_map)
    chosen = select(pairs, residue_types=residue_types, top_n=args.top_n,
                    top_n_intra=args.top_n_intra, min_seq_sep=args.min_seq_sep,
                    bead_key=bead_key)
    if not chosen:
        parser.error("no restraints selected -- widen --residue-types or raise --top-n")

    required_keys = {
        (pair.protein1, pair.copy1) for pair in chosen
    } | {
        (pair.protein2, pair.copy2) for pair in chosen
    }
    available_keys = set(atoms.keys())
    missing_keys = sorted(required_keys - available_keys)
    if missing_keys:
        provided = sorted(set(chains.values()))
        parser.error(
            "selected contacts require protein/copy mappings that are not present in "
            f"the provided --chains mapping/PDB: missing {missing_keys}. "
            f"Provided protein/copy mappings: {provided}. "
            "Check --chains (for example, if copy 1 of ECOIL is needed, include a "
            "chain mapped as ...=ECOIL:1)."
        )

    if args.wildcard_copies:
        crossing = [p for p in chosen if p.copy1 != p.copy2]
        if crossing:
            parser.error(
                f"--wildcard-copies is invalid: {len(crossing)} selected pair(s) cross "
                "copies, and '*' can only express copy-for-copy restraints")

    constraints: List[DistanceConstraint] = []
    print(f"{len(chosen)} restraint(s) selected from {len(pairs)} contact(s)")
    for pair in chosen:
        residues1 = beads[pair.protein1][pair.residue1]
        residues2 = beads[pair.protein2][pair.residue2]
        target = float(np.linalg.norm(
            bead_center(atoms[(pair.protein1, pair.copy1)], residues1)
            - bead_center(atoms[(pair.protein2, pair.copy2)], residues2)))
        flexible = "*" if pair.is_flexible else " "
        print(f"  {flexible} {pair.protein1}.{pair.copy1} {pair.residue1:>3d}"
              f"{pair.restype1} -- {pair.protein2}.{pair.copy2} {pair.residue2:>3d}"
              f"{pair.restype2} : residue {pair.distance:6.2f} A -> bead "
              f"{target:6.2f} A  [{pair.kind}]")
        constraints.append(DistanceConstraint(
            residue1=pair.residue1, protein1=pair.protein1,
            copy1=ANY_COPY if args.wildcard_copies else pair.copy1,
            residue2=pair.residue2, protein2=pair.protein2,
            copy2=ANY_COPY if args.wildcard_copies else pair.copy2,
            distance=target, force_constant=args.force_constant,
        ))

    write_distance_constraints(output, constraints)
    print(f"\nwrote {len(constraints)} restraint(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
