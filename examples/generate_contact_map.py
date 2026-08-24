"""Build a contact map from a ground-truth structure.

Run this **once per ground-truth structure** -- one per copy number, whether
hand-built or predicted.  The result is durable metadata: every residue pair
within a generous cutoff, with residue types, rigid-body labels and
residue-to-residue distances, ranked closest first.  Restraint sets are then
selected out of it at run time by generate_distance_restraints.py, so varying
"how many restraints, of what chemistry" costs a file read instead of
recomputing anything.

Nothing here builds a PMI system.  The only representation-dependent field is
the rigid-body label, and that comes from the `domains` list in each protein's
JSON, which is an input to the build rather than a product of it.  So a contact
map can be generated for a structure long before deciding how to coarse-grain
it, and stays valid if the bead size changes -- the distances are between
residue centroids, not beads.

Chain mapping
-------------
A predicted structure will use its own chain letters and say nothing about
which chain is which copy of which protein, so the mapping is declared
explicitly:

    python generate_contact_map.py --structure data/pdb/kcoil_ecoil.pdb \\
        --chains A=KCOIL:0 B=ECOIL:0 \\
        --output data/contact_maps/kcoil_ecoil_n1.csv

For a two-copy prediction that would be
`--chains A=KCOIL:0 B=ECOIL:0 C=KCOIL:1 D=ECOIL:1`.
"""

import argparse
import itertools
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kcoil_ecoil_system as system_builder
from impjax_toymodels.contact_map import (
    FLEXIBLE,
    ContactPair,
    classify,
    write_contact_map,
)

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

#: Wider than any crosslinker span, so nothing a restraint could ever use is
#: lost, while the file stays bounded for large assemblies.
DEFAULT_CUTOFF = 35.0

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_chain_spec(spec: str) -> Tuple[str, str, int]:
    """Parse one `CHAIN=PROTEIN:COPY` argument."""
    try:
        chain, rest = spec.split("=", 1)
        protein, copy_index = rest.rsplit(":", 1)
        return chain.strip(), protein.strip(), int(copy_index)
    except ValueError:
        raise SystemExit(
            f"bad --chains entry {spec!r}; expected CHAIN=PROTEIN:COPY, e.g. A=KCOIL:0"
        ) from None


def read_residues(pdb_path: str, chains: Dict[str, Tuple[str, int]]) -> List[dict]:
    """Every mapped residue's centroid and one-letter type, from the PDB.

    A plain unweighted mean over the residue's atoms, matching how IMP places
    a coarse bead, so residue-level and bead-level geometry stay consistent.
    """
    atoms: Dict[tuple, List[List[float]]] = {}
    names: Dict[tuple, str] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain not in chains:
                continue
            protein, copy_index = chains[chain]
            key = (protein, copy_index, int(line[22:26]))
            atoms.setdefault(key, []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
            names.setdefault(key, line[17:20].strip())

    residues = []
    for (protein, copy_index, residue), coords in atoms.items():
        residues.append({
            "protein": protein, "copy": copy_index, "residue": residue,
            "restype": THREE_TO_ONE.get(names[(protein, copy_index, residue)], "X"),
            "center": np.mean(coords, axis=0),
        })
    residues.sort(key=lambda r: (r["protein"], r["copy"], r["residue"]))
    return residues


def body_labels(protein: str, copy_index: int, data_dir: str) -> Dict[int, str]:
    """Map each residue to its rigid-body label, or FLEXIBLE if it has none.

    Read from the protein's JSON `domains`, the same list the build uses to
    decide which residues become rigid bodies.
    """
    info = system_builder._load(
        os.path.join(data_dir, "data", "json_files", f"{protein}.json"))
    labels: Dict[int, str] = {}
    for low, high in info["domains"]:
        for residue in range(low, high + 1):
            labels[residue] = f"{protein}_{copy_index}_{low}-{high}"
    return labels


def build_pairs(residues: List[dict], labels: Dict[tuple, Dict[int, str]],
                cutoff: float) -> List[ContactPair]:
    """Every residue pair within `cutoff`, ranked closest first."""
    found = []
    for first, second in itertools.combinations(residues, 2):
        distance = float(np.linalg.norm(first["center"] - second["center"]))
        if distance > cutoff:
            continue
        found.append((distance, first, second))
    found.sort(key=lambda item: item[0])

    pairs = []
    for rank, (distance, first, second) in enumerate(found, start=1):
        body1 = labels[(first["protein"], first["copy"])].get(first["residue"], FLEXIBLE)
        body2 = labels[(second["protein"], second["copy"])].get(second["residue"], FLEXIBLE)
        pairs.append(ContactPair(
            rank=rank,
            residue1=first["residue"], protein1=first["protein"],
            copy1=first["copy"], restype1=first["restype"], body1=body1,
            residue2=second["residue"], protein2=second["protein"],
            copy2=second["copy"], restype2=second["restype"], body2=body2,
            distance=distance,
            kind=classify(first["protein"], first["copy"],
                          second["protein"], second["copy"]),
        ))
    return pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--structure", required=True,
                        help="ground-truth PDB to derive the map from")
    parser.add_argument("--chains", nargs="+", required=True, metavar="CHAIN=PROTEIN:COPY",
                        help="which chain is which copy of which protein, "
                             "e.g. A=KCOIL:0 B=ECOIL:0")
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF,
                        help="keep residue pairs closer than this, in A "
                             "(default: %(default)s -- wider than any crosslinker span)")
    parser.add_argument("--data-dir", default=EXAMPLES_DIR,
                        help="base directory holding data/ (default: examples/)")
    parser.add_argument("--output", required=True, help="contact-map CSV to write")
    args = parser.parse_args(argv)

    chains = {}
    for spec in args.chains:
        chain, protein, copy_index = parse_chain_spec(spec)
        chains[chain] = (protein, copy_index)

    residues = read_residues(args.structure, chains)
    if not residues:
        parser.error(f"no mapped residues found in {args.structure}; check --chains")

    present = sorted({(r["protein"], r["copy"]) for r in residues})
    labels = {key: body_labels(key[0], key[1], args.data_dir) for key in present}
    print(f"{len(residues)} residues across {len(present)} chain(s): "
          + ", ".join(f"{p} copy {c}" for p, c in present))

    pairs = build_pairs(residues, labels, args.cutoff)
    by_kind: Dict[str, int] = {}
    for pair in pairs:
        by_kind[pair.kind] = by_kind.get(pair.kind, 0) + 1
    print(f"{len(pairs)} pair(s) within {args.cutoff} A")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:12s} {count:7d}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_contact_map(args.output, pairs)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
