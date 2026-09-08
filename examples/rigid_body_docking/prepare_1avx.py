"""Turn the 1AVX crystal structure into inputs a PMI system can be built from.

1AVX is porcine pancreatic trypsin (chain A) bound to soybean trypsin inhibitor
(chain B) -- the classic rigid-body docking benchmark target, and the ground
truth this example tries to recover from sparse synthetic crosslinks.

Why a preparation step is needed at all
---------------------------------------
The deposited file cannot be handed to PMI as it stands.  Trypsin is numbered
by alignment to chymotrypsinogen, which means:

* three residues carry **insertion codes** (184A, 188A, 221A).  IMP's PDB
  reader keeps the residue index and drops the code, so `IMP.atom.read_pdb`
  returns 224 residues with three *duplicate* indexes.  PMI's `add_structure`
  keys its residues on that index, so the duplicates silently collide and the
  built system is wrong -- with no error anywhere.
* six places where the numbering jumps (34->37, 67->69, 125->127, 130->132,
  204->209, 217->219).  These are **not** missing residues: every C-N bond
  across them measures under 1.45 A, so the chain is chemically continuous.
  They are deletions relative to chymotrypsinogen, an artifact of the
  numbering scheme.

Chain B is numbered 501-677 and has exactly one genuine break: residues
639-643 (QAEDD) are unobserved, with C638-N644 spanning 11.41 A.  That is the
only unmodelled region in the whole complex -- 5 residues out of 400.

So: both chains are renumbered contiguously from 1, folding insertion codes
into the sequence, and each chain's FASTA is derived from the residues that
are actually present.  `data/residue_mapping.csv` records the original
numbering so any restraint can be traced back to the deposited file.

Observed-only versus modelled gaps
----------------------------------
By default only observed residues get a slot, so every residue is structured,
PMI creates no flexible beads, and the system is exactly two rigid bodies --
which is the docking problem this example is about.

`--model-gaps` instead allocates slots for the residues REMARK 465 declares
missing at each genuine chain break, so they become coarse beads and the
system exercises the mixed rigid/flexible path.  Breaks are detected from the
geometry (C-N > 1.45 A) rather than from the numbering, which is what keeps
trypsin's six numbering artifacts from being mistaken for gaps.

Usage
-----
    python prepare_1avx.py                    # fetch from RCSB, observed only
    python prepare_1avx.py --pdb 1AVX.pdb     # use a local copy
    python prepare_1avx.py --model-gaps       # bead the 5 unobserved residues
"""

import argparse
import csv
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

RCSB_URL = "https://files.rcsb.org/download/{code}.pdb"

#: Longest plausible peptide C-N bond.  Anything above this is a real break in
#: the chain, not a jump in the numbering.
PEPTIDE_BOND_MAX = 1.45

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class ChainSpec:
    """How one deposited chain becomes one PMI molecule."""

    chain: str          # chain identifier in the deposited file
    protein: str        # molecule name used throughout the example
    description: str
    color: str          # RMF display colour, matching the JSON schema


CHAINS = (
    ChainSpec("A", "TRYP", "porcine pancreatic trypsin", "#4f7ff7"),
    ChainSpec("B", "STI", "soybean trypsin inhibitor (Kunitz)", "#f7754f"),
)


@dataclass
class Slot:
    """One position in the prepared sequence.

    An *observed* slot has coordinates and keeps the ATOM lines it came from;
    an unobserved slot (only ever created by --model-gaps) carries just a
    residue type, and PMI represents it as a coarse bead.
    """

    restype: str                     # three-letter residue name
    original: Optional[Tuple[int, str]] = None   # (resSeq, iCode) as deposited
    lines: List[str] = field(default_factory=list)

    @property
    def observed(self) -> bool:
        return self.original is not None

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.restype, "X")


def fetch(code: str, destination: str) -> str:
    """Download a PDB entry, unless it is already cached."""
    if os.path.exists(destination):
        return destination
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    url = RCSB_URL.format(code=code.upper())
    print(f"fetching {url}")
    urllib.request.urlretrieve(url, destination)
    return destination


def read_missing_residues(path: str) -> Dict[str, List[Tuple[int, str]]]:
    """Residues REMARK 465 declares unobserved, as {chain: [(resSeq, resname)]}.

    Only the residue records are parsed; the surrounding explanatory lines are
    identified by not matching the fixed column layout of a residue entry.
    """
    missing: Dict[str, List[Tuple[int, str]]] = {}
    for line in open(path):
        if not line.startswith("REMARK 465"):
            continue
        body = line[11:].rstrip()
        parts = body.split()
        # A residue record is "RES C SSSEQI", optionally preceded by a model
        # number; anything else on these lines is prose or the column header.
        if len(parts) < 3 or parts[0] not in THREE_TO_ONE:
            continue
        try:
            resseq = int(parts[2])
        except ValueError:
            continue
        missing.setdefault(parts[1], []).append((resseq, parts[0]))
    return missing


def read_chain(path: str, chain: str) -> List[Slot]:
    """Every observed residue of one chain, in file order, as slots.

    HETATM records are ignored, which removes the waters and the Ca(2+) ion in
    chain A: neither is part of either protein's sequence, and a bound ion
    inside a rigid body would only add a particle that never moves relative to
    the rest of it.  Hydrogens and minor altlocs are dropped so that bead
    centres, which are plain means over a residue's atoms, are reproducible.
    """
    slots: List[Slot] = []
    index: Dict[Tuple[int, str], Slot] = {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        if element == "H":
            continue
        key = (int(line[22:26]), line[26].strip())
        slot = index.get(key)
        if slot is None:
            slot = Slot(restype=line[17:20].strip(), original=key)
            index[key] = slot
            slots.append(slot)
        # Blank the altloc so the emitted file has a single, unambiguous copy.
        slot.lines.append(line[:16] + " " + line[17:])
    return slots


def atom_coordinate(slot: Slot, name: str) -> Optional[np.ndarray]:
    """Coordinates of one named backbone atom, or None if it is absent."""
    for line in slot.lines:
        if line[12:16].strip() == name:
            return np.array([float(line[30 + 8 * i: 38 + 8 * i]) for i in range(3)])
    return None


def find_breaks(slots: List[Slot]) -> List[int]:
    """Indexes i such that a real chain break lies between slot i and i+1.

    Measured from the peptide bond rather than inferred from the numbering:
    trypsin's numbering jumps six times without the chain ever breaking, and
    treating those as gaps would insert ten residues that are actually present.
    """
    breaks = []
    for i in range(len(slots) - 1):
        carbon = atom_coordinate(slots[i], "C")
        nitrogen = atom_coordinate(slots[i + 1], "N")
        if carbon is None or nitrogen is None:
            continue
        if float(np.linalg.norm(carbon - nitrogen)) > PEPTIDE_BOND_MAX:
            breaks.append(i)
    return breaks


def insert_missing(slots: List[Slot], breaks: List[int],
                   missing: List[Tuple[int, str]]) -> List[Slot]:
    """Add unobserved slots at each real break, from the REMARK 465 list.

    A break's missing residues are those whose deposited numbers fall strictly
    between the two flanking observed residues, which is unambiguous because a
    break is by definition a discontinuity in that numbering.
    """
    if not breaks:
        return slots

    filled: List[Slot] = []
    break_set = set(breaks)
    for i, slot in enumerate(slots):
        filled.append(slot)
        if i not in break_set:
            continue
        low = slots[i].original[0]
        high = slots[i + 1].original[0]
        gap = [(number, name) for number, name in missing if low < number < high]
        for number, name in sorted(gap):
            filled.append(Slot(restype=name, original=None))
        numbers = [str(number) for number, _ in gap]
        print(f"    break after residue {low}: modelling {len(gap)} unobserved "
              f"residue(s) {'-'.join(numbers[:1] + numbers[-1:])} as beads")
    return filled


def structured_ranges(slots: List[Slot]) -> List[Tuple[int, int]]:
    """Maximal runs of observed slots, as 1-based inclusive residue ranges.

    These become the rigid-body domains: a run of residues whose coordinates
    are known and which therefore move together.
    """
    ranges: List[Tuple[int, int]] = []
    start = None
    for position, slot in enumerate(slots, start=1):
        if slot.observed and start is None:
            start = position
        elif not slot.observed and start is not None:
            ranges.append((start, position - 1))
            start = None
    if start is not None:
        ranges.append((start, len(slots)))
    return ranges


def write_chain_pdb(path: str, slots: List[Slot], chain: str) -> None:
    """Emit one chain, renumbered contiguously with no insertion codes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(_chain_records(slots, chain, serial=1))
        handle.write("END\n")


def _chain_records(slots: List[Slot], chain: str, serial: int) -> str:
    """ATOM records for one chain, with new residue numbers and atom serials.

    Column-spliced rather than reformatted, so every field this script does not
    deliberately change (element, occupancy, B-factor, coordinates) survives
    exactly as deposited.
    """
    out = []
    for position, slot in enumerate(slots, start=1):
        if not slot.observed:
            continue  # no coordinates to write
        for line in slot.lines:
            out.append(
                f"{line[:6]}{serial:5d}{line[11:21]}{chain}{position:4d} {line[27:]}")
            serial += 1
    out.append(f"TER   {serial:5d}\n")
    return "".join(out)


def write_complex_pdb(path: str, chains: List[Tuple[str, List[Slot]]]) -> None:
    """Both chains in their crystallographic frame: the ground-truth complex."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        for chain, slots in chains:
            handle.write(_chain_records(slots, chain, serial=1))
        handle.write("END\n")


def write_fasta(path: str, protein: str, slots: List[Slot]) -> None:
    """One-letter sequence over every slot, observed or not.

    The "sp|" header is what `IMP.pmi.topology.Sequences` keys on, matching the
    convention the other example systems use.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sequence = "".join(slot.one_letter for slot in slots)
    with open(path, "w") as handle:
        handle.write(f">sp|{protein}\n")
        for start in range(0, len(sequence), 60):
            handle.write(sequence[start:start + 60] + "\n")


def write_json(path: str, spec: ChainSpec, slots: List[Slot]) -> None:
    """The per-protein build description, in the schema the examples share."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    document = {
        "uniprot_id": spec.protein,
        "protein_name": spec.protein,
        "description": spec.description,
        "files": {
            "pdb": f"data/pdb/{spec.protein}.pdb",
            "fasta": f"data/fasta/{spec.protein}.fasta",
        },
        # One rigid body per run of observed residues.  With no modelled gaps
        # that is the whole chain, which is what makes this a pure rigid-body
        # docking problem.
        "domains": [list(pair) for pair in structured_ranges(slots)],
        "oligomerization": False,
        "monomer_chain": [spec.chain],
        "degrees_of_freedom": "domains",
        "representation": {"structured": 2, "unstructured": 1},
        "visualization": {"color": spec.color},
    }
    with open(path, "w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def write_mapping(path: str, prepared: Dict[str, List[Slot]]) -> None:
    """Record original numbering against the new indexes, for traceability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein", "new_index", "restype",
                         "original_chain", "original_resseq", "original_icode",
                         "observed"])
        for spec in CHAINS:
            for position, slot in enumerate(prepared[spec.protein], start=1):
                resseq, icode = slot.original if slot.observed else ("", "")
                writer.writerow([spec.protein, position, slot.restype,
                                 spec.chain, resseq, icode,
                                 int(slot.observed)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdb", default=None,
                        help="local 1AVX PDB file (default: fetch from RCSB into "
                             "data/raw/1AVX.pdb)")
    parser.add_argument("--model-gaps", action="store_true",
                        help="give unobserved residues coarse beads instead of "
                             "splicing them out")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="where to write the prepared inputs (default: %(default)s)")
    args = parser.parse_args(argv)

    source = args.pdb or fetch("1AVX", os.path.join(args.data_dir, "raw", "1AVX.pdb"))
    missing = read_missing_residues(source)

    prepared: Dict[str, List[Slot]] = {}
    for spec in CHAINS:
        slots = read_chain(source, spec.chain)
        if not slots:
            parser.error(f"no ATOM records for chain {spec.chain} in {source}")
        breaks = find_breaks(slots)
        print(f"  chain {spec.chain} -> {spec.protein}: {len(slots)} observed residue(s), "
              f"{len(breaks)} real chain break(s)")
        if args.model_gaps:
            slots = insert_missing(slots, breaks, missing.get(spec.chain, []))
        elif breaks:
            print(f"    splicing {len(breaks)} break(s) out; pass --model-gaps to "
                  "represent the unobserved residues as beads instead")
        prepared[spec.protein] = slots

    for spec in CHAINS:
        slots = prepared[spec.protein]
        write_chain_pdb(os.path.join(args.data_dir, "pdb", f"{spec.protein}.pdb"),
                        slots, spec.chain)
        write_fasta(os.path.join(args.data_dir, "fasta", f"{spec.protein}.fasta"),
                    spec.protein, slots)
        write_json(os.path.join(args.data_dir, "json_files", f"{spec.protein}.json"),
                   spec, slots)
        ranges = structured_ranges(slots)
        print(f"  {spec.protein}: {len(slots)} residue(s), rigid-body domains {ranges}")

    write_complex_pdb(
        os.path.join(args.data_dir, "pdb", "1avx_complex.pdb"),
        [(spec.chain, prepared[spec.protein]) for spec in CHAINS])
    write_mapping(os.path.join(args.data_dir, "residue_mapping.csv"), prepared)

    print(f"\nwrote prepared inputs to {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
