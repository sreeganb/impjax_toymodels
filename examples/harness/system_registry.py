"""Resolve a `--system NAME` argument to the module that builds that system.

Everything else in examples/harness/ is system-agnostic: it generates contact
maps, selects restraints, runs samplers, measures accuracy and renders a
report, without knowing what molecules are involved.  The one thing it cannot
avoid knowing is *how to build the IMP system*, and that is exactly what a
system module supplies.

The split mirrors the one already enforced between src/ and examples/:
src/impjax_toymodels/ never mentions a specific system, examples/harness/
never mentions a specific system, and each system module mentions exactly one.
Adding a third system is a new module plus a config, not a fork of the harness.

The system-module protocol
--------------------------
A system module is an ordinary Python module exposing:

    PROTEINS : tuple[str, ...]
        Molecule names, in the fixed order the harness iterates them.  Bead
        ordering everywhere downstream follows this order, so it must not
        change between a reference build and a trajectory build.

    DATA_DIR : str
        Base directory whose "data" subfolder holds this system's inputs
        (json_files/, pdb/, fasta/, contact_maps/).  Same meaning as the
        `data_dir` parameter of the build functions below.

    DEFAULT_DISTANCE_CSV : str
        Default harmonic-restraint file, relative to DATA_DIR.

    build_system(copy_number, data_dir, shuffle, distance_csv)
        -> (built, score_function, output_objects)
        One combined scoring function over every restraint.

    build_split(copy_number, data_dir, shuffle, distance_csv)
        -> (built, likelihood_sf, prior_sf, output_objects)
        The same system with its restraints partitioned into a tempered
        likelihood and an untempered prior.  The two scoring functions must be
        disjoint, or the posterior double-counts at lambda = 1 (priors.
        restraint_prior checks this and raises).  Which restraints go where is
        a per-system judgement: for a chain-connected system connectivity is
        the natural prior, while for a rigid-body docking problem connectivity
        is a constant and excluded volume is the meaningful structural prior.

    domains_for(protein) -> list[(low, high)]
        Residue ranges that become rigid bodies, used to label residues in a
        contact map.  An input to the build, not a product of it, so a contact
        map can be generated before any system is built.

Modules are looked up by name in examples/ and in each examples/*/ package
directory, so `--system kcoil_ecoil_system` and `--system docking_system` both
resolve without the caller knowing where the file lives.
"""

import importlib
import os
import sys
from types import ModuleType
from typing import List

#: examples/, the parent of this harness package.
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Used when no --system is given, so every pre-existing command line that
#: predates the harness split keeps working unchanged.
DEFAULT_SYSTEM = "kcoil_ecoil_system"

#: Names a module must define to satisfy the protocol documented above.
REQUIRED_ATTRIBUTES = (
    "PROTEINS",
    "DATA_DIR",
    "DEFAULT_DISTANCE_CSV",
    "build_system",
    "build_split",
    "domains_for",
)


def search_paths() -> List[str]:
    """Directories a system module may live in: examples/ and its subpackages.

    Subdirectories are included so a system can own a folder holding its own
    data/, README and system-specific analysis code (as
    examples/rigid_body_docking/ does) rather than scattering those through
    examples/ alongside every other system's.
    """
    paths = [EXAMPLES_DIR]
    for entry in sorted(os.listdir(EXAMPLES_DIR)):
        candidate = os.path.join(EXAMPLES_DIR, entry)
        # "data" and "out" hold inputs and results, never importable modules;
        # skipping them keeps a stray *.py in an output tree from shadowing a
        # real system module.
        if entry in ("data", "out", "harness", "__pycache__") or entry.startswith("."):
            continue
        if os.path.isdir(candidate):
            paths.append(candidate)
    return paths


def available() -> List[str]:
    """Every module name on the search path that satisfies the protocol.

    Imports each candidate to check it, which is unavoidable -- the protocol is
    about module attributes, not filenames -- but only ever runs on an explicit
    request for the list (a bad --system, or --list-systems), never on the
    happy path.
    """
    found = []
    for directory in search_paths():
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            name = filename[: -len(".py")]
            try:
                module = _import(name)
            except Exception:
                continue  # not importable, so not a usable system module
            if all(hasattr(module, attribute) for attribute in REQUIRED_ATTRIBUTES):
                found.append(name)
    return sorted(set(found))


def _import(name: str) -> ModuleType:
    """Import `name` with examples/ and its subpackages on sys.path."""
    for directory in search_paths():
        if directory not in sys.path:
            sys.path.insert(0, directory)
    return importlib.import_module(name)


def resolve(name: str = None) -> ModuleType:
    """Import the named system module and check it against the protocol.

    Raising here, with the list of what is actually available, is the whole
    point: without it a typo'd --system surfaces three frames deep as an
    AttributeError on a half-imported module.
    """
    name = name or DEFAULT_SYSTEM
    try:
        module = _import(name)
    except ImportError as error:
        raise SystemExit(
            f"unknown system {name!r} ({error}).\n"
            f"available: {', '.join(available()) or '(none found)'}"
        ) from None

    missing = [a for a in REQUIRED_ATTRIBUTES if not hasattr(module, a)]
    if missing:
        raise SystemExit(
            f"module {name!r} ({module.__file__}) is not a system module: "
            f"missing {', '.join(missing)}.\n"
            "See examples/harness/system_registry.py for the protocol."
        )
    return module


def add_argument(parser) -> None:
    """Add the standard --system option to an argparse parser.

    Defined once so every harness script spells the option and its help text
    identically.
    """
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="system module to build (default: %(default)s)",
    )
