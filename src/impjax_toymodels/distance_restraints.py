"""Harmonic distance restraints driven by a CSV constraint file.

This module is deliberately system-agnostic: it knows nothing about KCOIL,
ECOIL or any other particular complex.  It only knows how to turn rows of a
CSV file into `IMP.pmi.restraints.basic.DistanceRestraint` objects attached to
whatever hierarchy it is handed, which is what lets one restraint file be
reused across different systems, copy numbers and samplers.

CSV format
----------
One restraint per row, with the header::

    residue1,protein1,copy1,residue2,protein2,copy2,distance,force_constant

* ``residue1`` / ``residue2`` -- 1-based residue indexes.  At a coarse-grained
  representation a residue index selects the *bead that contains* that residue,
  so several residue indexes can name the same particle; the generator that
  writes these files therefore emits one representative residue per bead.
* ``protein1`` / ``protein2`` -- PMI molecule names (``IMP.atom.Selection``'s
  ``molecule=`` argument).
* ``copy1`` / ``copy2`` -- 0-based PMI copy indexes, or ``*``.  A restraint is
  usually a statement about one assembly, and when the model contains N copies
  of that assembly the same statement holds inside each of them.  Writing ``*``
  in *both* copy columns expresses exactly that: at load time the row is
  expanded into one restraint per copy index, pairing copy i with copy i.
  Explicit integers are still allowed (and are what the generator writes by
  default) for restraints that genuinely single out one copy, or that cross
  from one copy to another.  Mixing ``*`` with an integer is rejected, because
  "copy 2 of protein A to *every* copy of protein B" is a different (and much
  less common) statement that should be spelled out row by row.
* ``distance`` -- the harmonic's equilibrium distance, in angstroms.
* ``force_constant`` -- the harmonic's kappa, in kcal/mol/A^2.

Each row becomes a *true* harmonic well.  PMI's ``DistanceRestraint`` is a
flat-bottomed restraint built from a `HarmonicUpperBound` at ``distancemax``
and a `HarmonicLowerBound` at ``distancemin``; passing the same value for both
collapses the flat bottom to a point, leaving the two one-sided harmonics to
cover opposite sides of the same minimum, i.e. score = 0.5 * kappa * (d - d0)^2.
"""

import contextlib
import csv
import io
import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import IMP.atom
import IMP.pmi.restraints.basic

logger = logging.getLogger(__name__)

#: Column order of the constraint CSV, used for both reading and writing.
COLUMNS = (
    "residue1", "protein1", "copy1",
    "residue2", "protein2", "copy2",
    "distance", "force_constant",
)

#: Value in a copy column meaning "every copy, paired copy-for-copy".
ANY_COPY = "*"

CopyIndex = Union[int, str]


@dataclass(frozen=True)
class DistanceConstraint:
    """One row of the constraint CSV: a harmonic between two coarse beads."""

    residue1: int
    protein1: str
    copy1: CopyIndex
    residue2: int
    protein2: str
    copy2: CopyIndex
    distance: float
    force_constant: float

    def as_row(self) -> dict:
        """Render back to CSV form, so a read/write round trip is lossless."""
        return {
            "residue1": self.residue1, "protein1": self.protein1, "copy1": self.copy1,
            "residue2": self.residue2, "protein2": self.protein2, "copy2": self.copy2,
            "distance": f"{self.distance:.3f}",
            "force_constant": f"{self.force_constant:.3f}",
        }

    @property
    def label(self) -> str:
        """Unique, human-readable name used for the restraint's stat-file key."""
        return (f"{self.protein1}.{self.copy1}.{self.residue1}"
                f"_{self.protein2}.{self.copy2}.{self.residue2}")


def _parse_copy(raw: str, row_number: int) -> CopyIndex:
    """Accept either ``*`` or a non-negative integer in a copy column."""
    value = raw.strip()
    if value == ANY_COPY:
        return ANY_COPY
    try:
        index = int(value)
    except ValueError:
        raise ValueError(
            f"row {row_number}: copy index must be an integer or '{ANY_COPY}', got {raw!r}"
        ) from None
    if index < 0:
        raise ValueError(f"row {row_number}: copy index must be >= 0, got {index}")
    return index


def read_distance_constraints(path: str) -> List[DistanceConstraint]:
    """Read a constraint CSV into `DistanceConstraint` records, unexpanded.

    Copy wildcards are preserved here rather than resolved, because resolving
    them needs a copy number that only the caller (which built the system)
    knows.  Call `expand_copies` next.
    """
    constraints: List[DistanceConstraint] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing required column(s): {', '.join(sorted(missing))}"
            )
        # Row 1 is the header, so data rows are numbered from 2 -- matching what
        # a text editor shows, which is what makes an error message actionable.
        for row_number, row in enumerate(reader, start=2):
            if not any((row.get(column) or "").strip() for column in COLUMNS):
                continue  # tolerate blank separator lines
            copy1 = _parse_copy(row["copy1"], row_number)
            copy2 = _parse_copy(row["copy2"], row_number)
            if (copy1 == ANY_COPY) != (copy2 == ANY_COPY):
                raise ValueError(
                    f"row {row_number}: '{ANY_COPY}' must appear in both copy columns "
                    "or neither; write per-copy rows explicitly instead"
                )
            force_constant = float(row["force_constant"])
            if force_constant <= 0.0:
                raise ValueError(
                    f"row {row_number}: force_constant must be > 0, got {force_constant}"
                )
            constraints.append(DistanceConstraint(
                residue1=int(row["residue1"]),
                protein1=row["protein1"].strip(),
                copy1=copy1,
                residue2=int(row["residue2"]),
                protein2=row["protein2"].strip(),
                copy2=copy2,
                distance=float(row["distance"]),
                force_constant=force_constant,
            ))
    return constraints


def write_distance_constraints(path: str, constraints: Iterable[DistanceConstraint]) -> None:
    """Write `DistanceConstraint` records out in the canonical column order."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for constraint in constraints:
            writer.writerow(constraint.as_row())


def expand_copies(
    constraints: Sequence[DistanceConstraint], copy_number: int
) -> List[DistanceConstraint]:
    """Resolve ``*`` copy wildcards against a concrete copy count.

    Rows with explicit copy indexes pass through untouched but are validated:
    a restraint naming copy 3 of a system built with two copies is a silent
    no-op at best and a wrong-particle restraint at worst, so it is an error.
    """
    expanded: List[DistanceConstraint] = []
    for constraint in constraints:
        if constraint.copy1 == ANY_COPY:
            for copy_index in range(copy_number):
                expanded.append(DistanceConstraint(
                    **{**constraint.__dict__, "copy1": copy_index, "copy2": copy_index}
                ))
            continue
        for copy_index in (constraint.copy1, constraint.copy2):
            if copy_index >= copy_number:
                raise ValueError(
                    f"constraint {constraint.label} names copy {copy_index}, but the "
                    f"system was built with only {copy_number} copies (0-{copy_number - 1})"
                )
        expanded.append(constraint)
    return expanded


def add_distance_restraints(
    root_hier,
    constraints: Sequence[DistanceConstraint],
    resolution: float = 1.0,
    weight: float = 1.0,
    verbose: bool = False,
) -> List:
    """Build one harmonic PMI restraint per constraint and add it to the model.

    Parameters
    ----------
    root_hier : the built PMI hierarchy to select particles from.
    constraints : copy-expanded constraints (see `expand_copies`).
    resolution : bead resolution to select at.  IMP picks the represented
        resolution closest to this value, so 1.0 resolves to the finest
        available bead for each residue -- per-residue beads in unstructured
        regions, and the enclosing coarse fragment inside structured domains.
    weight : multiplies every restraint's score; the per-restraint strength
        lives in the CSV's force_constant column, this scales the whole set.
    verbose : let PMI print its two lines per restraint.  Off by default:
        a realistic constraint file has hundreds of rows and, multiplied by
        the copy number, that chatter buries everything else the run says.
        A single summary line is logged instead.

    Returns
    -------
    list of PMI restraint objects (not raw `IMP.Restraint`s), already added to
    the model, so callers can both collect `.get_restraint()` for a scoring
    function and hand the objects to IMP's stat-file machinery.
    """
    restraints = []
    # PMI's DistanceRestraint prints unconditionally; swallow that rather than
    # letting it drown the log, but keep any real exception's context intact.
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with sink:
        for constraint in constraints:
            selection1 = (constraint.residue1, constraint.residue1,
                          constraint.protein1, constraint.copy1)
            selection2 = (constraint.residue2, constraint.residue2,
                          constraint.protein2, constraint.copy2)
            restraint = IMP.pmi.restraints.basic.DistanceRestraint(
                root_hier,
                selection1,
                selection2,
                # Equal min and max turn PMI's flat-bottomed well into a single
                # harmonic minimum at the ground-truth distance.
                distancemin=constraint.distance,
                distancemax=constraint.distance,
                resolution=resolution,
                kappa=constraint.force_constant,
                label=constraint.label,
                weight=weight,
            )
            restraint.add_to_model()
            restraints.append(restraint)
    logger.info("Added %d harmonic distance restraints at resolution %g",
                len(restraints), resolution)
    return restraints


def load_and_add(
    root_hier,
    path: str,
    copy_number: int,
    resolution: float = 1.0,
    weight: float = 1.0,
    verbose: bool = False,
) -> List:
    """Read, copy-expand and attach a constraint file in one call."""
    constraints = expand_copies(read_distance_constraints(path), copy_number)
    return add_distance_restraints(root_hier, constraints, resolution=resolution,
                                   weight=weight, verbose=verbose)
