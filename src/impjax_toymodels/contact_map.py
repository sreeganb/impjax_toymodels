"""Ground-truth contact maps, and the restraint selection that reads them.

A contact map is the exhaustive, *representation-independent* inventory of
residue pairs in a known structure: every pair within a generous cutoff, with
both residue types, both rigid-body labels, and the residue-to-residue
distance.  It is derived once per ground-truth structure and then reused.

Why the split matters
---------------------
Three things get conflated easily and should not be:

* the **ground-truth structure** -- an input (hand-built now, predicted later),
  one per copy number;
* the **contact map** -- metadata derived from it, expensive to compute,
  computed once, never varied per run;
* the **restraint set** -- a cheap selection from the map, changed constantly
  (how many, which chemistry, inter- or intra-molecular).

Keeping them apart means sweeping "how many restraints?" costs a file read
rather than a rebuild, and means an ensemble of ground-truth structures is
just several contact maps to combine, rather than a redesign.

Distances: residue-level here, bead-level there
-----------------------------------------------
The distances stored in a map are between residue centroids, so a map stays
valid when the coarse-graining changes -- it describes the structure, not the
representation.  They are used for *ranking* only.  The target distance
written into an actual restraint has to be measured between the beads the
restraint will act on, in whatever representation is in play, or the ground
truth will not sit at the restraint's minimum (two residues sharing a
resolution-2 bead differ from the bead center by up to ~0.5 A).  That second
measurement belongs to whoever builds the system; this module only ranks and
selects.
"""

import csv
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

#: Column order of the contact-map CSV.
COLUMNS = (
    "rank",
    "residue1", "protein1", "copy1", "restype1", "body1",
    "residue2", "protein2", "copy2", "restype2", "body2",
    "distance", "kind",
)

#: Label used in a body column for a residue outside every rigid body.
FLEXIBLE = "linker"

#: `kind` values, distinguishing the three ways a pair can span the assembly.
INTER_MOL = "inter_mol"    # two different proteins, same copy: one assembly
INTRA_MOL = "intra_mol"    # one protein, one copy: within a single chain
INTER_COPY = "inter_copy"  # different copies: across assemblies


@dataclass(frozen=True)
class ContactPair:
    """One row of a contact map: a residue pair in the ground-truth structure."""

    rank: int
    residue1: int
    protein1: str
    copy1: int
    restype1: str
    body1: str
    residue2: int
    protein2: str
    copy2: int
    restype2: str
    body2: str
    distance: float
    kind: str

    @property
    def is_flexible(self) -> bool:
        """True if either end lies outside a rigid body."""
        return self.body1 == FLEXIBLE or self.body2 == FLEXIBLE

    @property
    def same_body(self) -> bool:
        """True if both ends are in the *same* rigid body.

        Such a pair carries no information: its distance cannot change, so a
        restraint on it only adds a constant to the score.
        """
        return self.body1 != FLEXIBLE and self.body1 == self.body2

    def as_row(self) -> dict:
        return {
            "rank": self.rank,
            "residue1": self.residue1, "protein1": self.protein1,
            "copy1": self.copy1, "restype1": self.restype1, "body1": self.body1,
            "residue2": self.residue2, "protein2": self.protein2,
            "copy2": self.copy2, "restype2": self.restype2, "body2": self.body2,
            "distance": f"{self.distance:.3f}", "kind": self.kind,
        }


def classify(protein1: str, copy1: int, protein2: str, copy2: int) -> str:
    """Which of the three spanning relationships a pair represents."""
    if copy1 != copy2:
        return INTER_COPY
    return INTRA_MOL if protein1 == protein2 else INTER_MOL


def write_contact_map(path: str, pairs: Iterable[ContactPair]) -> None:
    """Write pairs out in the canonical column order."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.as_row())


def read_contact_map(path: str) -> List[ContactPair]:
    """Read a contact map, preserving its stored order (closest first)."""
    pairs: List[ContactPair] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing required column(s): {', '.join(sorted(missing))}")
        for row in reader:
            pairs.append(ContactPair(
                rank=int(row["rank"]),
                residue1=int(row["residue1"]), protein1=row["protein1"].strip(),
                copy1=int(row["copy1"]), restype1=row["restype1"].strip(),
                body1=row["body1"].strip(),
                residue2=int(row["residue2"]), protein2=row["protein2"].strip(),
                copy2=int(row["copy2"]), restype2=row["restype2"].strip(),
                body2=row["body2"].strip(),
                distance=float(row["distance"]), kind=row["kind"].strip(),
            ))
    return pairs


def select(
    pairs: Sequence[ContactPair],
    residue_types: Optional[Sequence[str]] = None,
    top_n: int = 10,
    top_n_intra: int = 2,
    min_seq_sep: int = 3,
    bead_key: Optional[Callable[[ContactPair], tuple]] = None,
) -> List[ContactPair]:
    """Pick a sparse, crosslink-like restraint set out of a contact map.

    Parameters
    ----------
    pairs : the contact map, as read.
    residue_types : one-letter codes the crosslinker reacts with; None keeps
        every pair regardless of residue identity.
    top_n : how many restraints to keep per assembly-spanning group -- that is,
        `top_n` for each distinct `inter_mol` protein pair and each distinct
        `inter_copy` group, taken closest-first.
    top_n_intra : how many `intra_mol` restraints to keep per chain.
    min_seq_sep : drop same-chain pairs closer than this many residues along
        the sequence; the connectivity restraint already covers them.
    bead_key : maps a pair to the identity of the two beads it would restrain.
        Several residues can share one coarse bead, and two restraints between
        the same bead pair are near-duplicates, so only the closest of each is
        kept.  None disables the deduplication.

    Returns
    -------
    The selected pairs, closest first.  Selection happens per group so that a
    single tight interface cannot consume the whole budget and leave another
    part of the assembly unrestrained.
    """
    reactive = set(residue_types) if residue_types else None

    groups: dict = {}
    for pair in pairs:
        if pair.same_body:
            continue  # frozen distance: no information
        if reactive is not None and not (
                pair.restype1 in reactive and pair.restype2 in reactive):
            continue
        if pair.kind == INTRA_MOL and abs(pair.residue1 - pair.residue2) < min_seq_sep:
            continue  # already covered by connectivity
        # One budget per (kind, participants) group.
        if pair.kind == INTRA_MOL:
            key = (INTRA_MOL, pair.protein1, pair.copy1)
        else:
            key = (pair.kind,
                   tuple(sorted(((pair.protein1, pair.copy1),
                                 (pair.protein2, pair.copy2)))))
        groups.setdefault(key, []).append(pair)

    selected: List[ContactPair] = []
    for key, candidates in groups.items():
        budget = top_n_intra if key[0] == INTRA_MOL else top_n
        candidates.sort(key=lambda p: p.distance)
        seen = set()
        taken = 0
        for pair in candidates:
            if taken >= budget:
                break
            if bead_key is not None:
                identity = bead_key(pair)
                if identity in seen:
                    continue  # same bead pair as one already taken
                seen.add(identity)
            selected.append(pair)
            taken += 1

    selected.sort(key=lambda p: p.distance)
    return selected
