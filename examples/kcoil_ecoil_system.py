"""Build the KCOIL/ECOIL toy coiled-coil system used by run_kcoil_ecoil_sampling.py.

This is system-specific IMP construction code, not source code: it belongs
in examples/ (and can exceed src/'s 200-300 line budget) precisely because
it is tied to one particular toy system. src/impjax_toymodels/ stays
system-agnostic -- it only ever consumes a `BuiltSystem` and an
`IMP.core.RestraintsScoringFunction`, however they were built. Swap this
file for a different system-building recipe to sample something else
through the same wrapper.

Mirrors test/test_imp_system.py's JSON + "domains" degrees-of-freedom path:
each protein (KCOIL, ECOIL) gets `copy_number` copies, each copy has two
structured domains (residues 1-21 and 32-52, each its own rigid body)
connected by a flexible unstructured linker (residues 22-31, flexible
beads).  Three restraint families score it: one connectivity restraint per
copy, one excluded-volume restraint over everything, and the harmonic
distance restraints listed in data/distance_constraints.csv -- which are
measured off the ground-truth complex by generate_distance_restraints.py,
so that sampling from a shuffled start has something to recover it with.

Two small, deliberate differences from the original script (both fixed
bugs, not behavior changes to the physical system): every copy's
connectivity restraint is actually collected into the returned scoring
function (the original only kept the last one -- the rest were added to
the IMP model but never scored), and each domain's rigid body gets a
unique name (the original reused one name across both domains of a copy).
"""

import json
import os
from typing import Dict, List, Tuple

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.dof
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.basic
import IMP.pmi.topology
from IMP.pmi.tools import OrderedSet

from impjax_toymodels import distance_restraints
from impjax_toymodels.system_info import BuiltSystem

EXAMPLES_DIR = os.path.dirname(__file__)
PROTEINS = ("KCOIL", "ECOIL")

#: Default constraint file, derived from the ground-truth complex by
#: generate_distance_restraints.py.  Relative to the resolved data dir.
DEFAULT_DISTANCE_CSV = os.path.join("data", "distance_constraints.csv")

MoleculeKey = Tuple[str, str]
MoleculesByKey = Dict[MoleculeKey, List]


def _load(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _build_component(state, json_path: str, data_dir: str, copy_number: int):
    """Create `copy_number` copies of one protein, each with its structured
    domains (from the PDB) and an unstructured linker (coarse beads)."""
    info = _load(json_path)

    if info["oligomerization"]:
        chain_letters = info["oligomerization_chains"]
        chains = chain_letters * (copy_number // len(chain_letters))
    else:
        chains = info["monomer_chain"] * copy_number

    fasta_path = os.path.join(data_dir, info["files"]["fasta"])
    pdb_path = os.path.join(data_dir, info["files"]["pdb"])
    sequences = IMP.pmi.topology.Sequences(fasta_path)

    root_mol = state.create_molecule(
        info["protein_name"], sequence=sequences["sp"], chain_id=chains[0]
    )

    mols = []
    for copy_index in range(copy_number):
        target = root_mol if copy_index == 0 else root_mol.create_copy(chain_id=chains[copy_index])
        atomic = OrderedSet()
        for domain in info["domains"]:
            structured = target.add_structure(
                pdb_path, chain_id=chains[copy_index], res_range=tuple(domain)
            )
            target.add_representation(
                structured,
                resolutions=info["representation"]["structured"],
                color=info["visualization"]["color"],
            )
            for particle in structured:
                atomic.add(particle)
        target.add_representation(
            target[:] - atomic,
            resolutions=info["representation"]["unstructured"],
            color=info["visualization"]["color"],
        )
        mols.append(target)

    return (info["uniprot_id"], info["protein_name"]), mols


def _build_rigid_bodies_and_flexible_beads(dof, root_hier, json_path: str, molecules: MoleculesByKey) -> None:
    """One rigid body per structured domain, flexible beads for the linker."""
    info = _load(json_path)
    key = (info["uniprot_id"], info["protein_name"])
    name = info["protein_name"]

    for mol in molecules[key]:
        copy_index = IMP.atom.Copy(mol.get_hierarchy()).get_copy_index()
        for domain in info["domains"]:
            selection = IMP.atom.Selection(
                root_hier,
                molecule=name,
                residue_indexes=range(domain[0], domain[1] + 1),
                copy_index=copy_index,
                resolution=IMP.atom.ALL_RESOLUTIONS,
            ).get_selected_particles()
            dof.create_rigid_body(selection, name=f"{name}_{copy_index}_{domain[0]}-{domain[1]}")
        dof.create_flexible_beads(mol.get_non_atomic_residues())


def _add_connectivity_restraints(molecules: MoleculesByKey) -> list:
    """One connectivity restraint per molecule copy, all returned (and hence
    scored) -- see module docstring for why this differs from the original
    script."""
    restraints = []
    for mols in molecules.values():
        for mol in mols:
            copy_index = IMP.atom.Copy(mol.get_hierarchy()).get_copy_index()
            connectivity = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
            connectivity.set_label(f"Connectivity.{mol.get_name()}.{copy_index}")
            connectivity.add_to_model()
#            restraints.append(connectivity.get_restraint())
            restraints.append(connectivity)
    return restraints

def _add_excluded_volume_restraints(root_hier):
    evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=root_hier,
                                                                    resolution=10)
    return evr

def _add_distance_restraints(root_hier, copy_number: int, csv_path: str) -> list:
    """Attach every harmonic distance restraint listed in a constraint CSV.

    The restraints themselves are not hand-written here any more: they are
    derived from the ground-truth complex by generate_distance_restraints.py
    and stored in data/distance_constraints.csv, which this function simply
    reads.  That keeps the *data* (which residue pairs, at what distance, with
    what force constant) out of the code, so a different restraint set --
    sparser, noisier, a different cutoff -- is a different CSV, not a different
    build script.

    Copy wildcards in the CSV are expanded against `copy_number` here, which is
    the only place that number is known.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"distance constraint file not found: {csv_path}\n"
            "Generate it from the ground-truth structure first:\n"
            "    python examples/generate_distance_restraints.py"
        )
    return distance_restraints.load_and_add(root_hier, csv_path, copy_number)


def _build_system_and_restraints(copy_number: int, data_dir: str,
                                 shuffle: bool = True, distance_csv: str = None):
    """Shared construction: build the system, then its two restraint families.

    Factored out so the system can be handed back either as one combined
    scoring function (`build_kcoil_ecoil_system`) or as a prior/likelihood
    partition (`build_kcoil_ecoil_split`), without building it twice or
    duplicating any of the topology code.

    Parameters
    ----------
    copy_number : number of copies of *each* protein to build.
    data_dir : base directory whose "data" subfolder holds the inputs.
    shuffle : randomize the starting configuration.  True is what sampling
        wants -- inference has to start away from the answer.  False leaves
        every rigid body on its PDB coordinates, i.e. at the ground truth,
        which is what generate_distance_restraints.py needs in order to read
        target distances off the reference structure, and what evaluate_
        recovery.py compares sampled models against.  Note this positions
        the rigid bodies only: the flexible linker has no structure read in,
        so PMI leaves its beads on a placeholder and nothing here moves them.
    distance_csv : path to the harmonic distance constraint file; defaults
        to DEFAULT_DISTANCE_CSV inside the resolved data dir.  Pass False to
        build the system with no distance restraints at all.

    Returns
    -------
    built : system_info.BuiltSystem
    connectivity : list of PMI connectivity restraint objects (one per copy)
    excluded_volume : the single PMI excluded-volume restraint object
    disres : list of PMI harmonic distance restraint objects, one per CSV row
    """
    base_dir = data_dir or EXAMPLES_DIR
    model = IMP.Model()
    system = IMP.pmi.topology.System(model)
    state = system.create_state()

    molecules: MoleculesByKey = {}
    for protein in PROTEINS:
        json_path = os.path.join(base_dir, "data", "json_files", f"{protein}.json")
        key, mols = _build_component(state, json_path, base_dir, copy_number)
        molecules[key] = mols

    root_hier = system.build()
    dof = IMP.pmi.dof.DegreesOfFreedom(model)
    for protein in PROTEINS:
        json_path = os.path.join(base_dir, "data", "json_files", f"{protein}.json")
        _build_rigid_bodies_and_flexible_beads(dof, root_hier, json_path, molecules)

    connectivity = _add_connectivity_restraints(molecules)
    excluded_volume = _add_excluded_volume_restraints(root_hier)
    if distance_csv is False:
        # Bootstrap path: generate_distance_restraints.py builds this very
        # system in order to *write* the constraint file, so it cannot also
        # require the file to already exist.
        disres = []
    else:
        csv_path = distance_csv or os.path.join(base_dir, DEFAULT_DISTANCE_CSV)
        disres = _add_distance_restraints(root_hier, copy_number, csv_path)

    # Inference has to start from a random configuration: if we started from
    # the PDB coordinates the answer would already be in hand.  Skipped only
    # when the caller explicitly asks for the reference state (see `shuffle`).
    if shuffle:
        mol_names = []
        for k, ms in molecules.items():
            mol_names += ms
        IMP.pmi.tools.shuffle_configuration(mol_names,
                                            max_translation=200,
                                            avoidcollision_rb=True)

    built = BuiltSystem(
        model=model,
        system=system,
        state=state,
        root_hier=root_hier,
        dof=dof,
        molecules=molecules,
    )
    return built, connectivity, excluded_volume, disres


def build_kcoil_ecoil_system(copy_number: int = 4, data_dir: str = None,
                             shuffle: bool = True, distance_csv: str = None):
    """Build the two-protein (KCOIL, ECOIL) toy coiled-coil system.

    Every restraint (connectivity + excluded volume + the harmonic distance
    restraints read from data/distance_constraints.csv) goes into a single
    scoring function, so the sampler treats the whole thing as one target
    with a flat prior. For the restraint-partitioned arrangement instead,
    see `build_kcoil_ecoil_split`.

    Parameters
    ----------
    copy_number : number of copies of *each* protein to build.
    data_dir : base directory whose "data" subfolder holds json_files/,
        pdb/, fasta/ and distance_constraints.csv (matching each JSON's own
        "data/..." relative paths, e.g. "data/fasta/KCOIL.fasta"); defaults
        to the examples/ directory itself, i.e. this file's own examples/data/.
    shuffle : randomize the starting configuration (True) or leave every
        rigid body on its ground-truth PDB coordinates (False).
    distance_csv : override the default constraint file path, or False to
        build with no distance restraints.

    Returns
    -------
    built : system_info.BuiltSystem
    score_function : IMP.core.RestraintsScoringFunction over all restraints
    output_objects : PMI restraint objects, for IMP's own stat-file machinery
    """
    built, connectivity, excluded_volume, disres = _build_system_and_restraints(
        copy_number, data_dir, shuffle=shuffle, distance_csv=distance_csv)

    output_objects = list(connectivity) + [excluded_volume] + disres
    restraints_set = [r.get_restraint() for r in connectivity] + [excluded_volume.get_restraint()] + [r.get_restraint() for r in disres]
    score_function = IMP.core.RestraintsScoringFunction(restraints_set)
    return built, score_function, output_objects


def build_kcoil_ecoil_split(copy_number: int = 4, data_dir: str = None,
                            shuffle: bool = True, distance_csv: str = None):
    """Build the same system with its restraints partitioned prior/likelihood.

    Connectivity goes into its own scoring function to be used as the prior
    (impjax_toymodels.priors.restraint_prior), excluded volume into another
    to be used as the tempered likelihood. The two must be disjoint or the
    posterior would double-count -- which is exactly what this split
    guarantees, and what priors.restraint_prior checks for.

    The distance restraints read from data/distance_constraints.csv are the
    experimental-data stand-in, so they go on the likelihood side together
    with excluded volume; connectivity is the structural prior.  (Distance
    restraints previously went into neither scoring function on this path --
    they were added to the IMP model but never scored, so the split sampler
    was inferring against excluded volume alone.)

    Returns
    -------
    built : system_info.BuiltSystem
    likelihood_score_function : excluded volume + distance restraints -- the
        tempered term
    prior_score_function : connectivity only -- the untempered structural prior
    output_objects : PMI restraint objects, for IMP's own stat-file machinery
    """
    built, connectivity, excluded_volume, disres = _build_system_and_restraints(
        copy_number, data_dir, shuffle=shuffle, distance_csv=distance_csv)

    likelihood_sf = IMP.core.RestraintsScoringFunction(
        [excluded_volume.get_restraint()]
        + [r.get_restraint() for r in disres])
    prior_sf = IMP.core.RestraintsScoringFunction([r.get_restraint() for r in connectivity])
    return built, likelihood_sf, prior_sf, list(connectivity) + [excluded_volume] + disres
