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
beads), scored by one connectivity restraint per copy.

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
import IMP.pmi.topology
from IMP.pmi.tools import OrderedSet

from impjax_toymodels.system_info import BuiltSystem

EXAMPLES_DIR = os.path.dirname(__file__)
PROTEINS = ("KCOIL", "ECOIL")

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
            restraints.append(connectivity.get_restraint())
    return restraints


def build_kcoil_ecoil_system(copy_number: int = 4, data_dir: str = None):
    """Build the two-protein (KCOIL, ECOIL) toy coiled-coil system.

    Parameters
    ----------
    copy_number : number of copies of *each* protein to build.
    data_dir : base directory whose "data" subfolder holds json_files/,
        pdb/, fasta/ (matching each JSON's own "data/..." relative paths,
        e.g. "data/fasta/KCOIL.fasta"); defaults to the examples/ directory
        itself, i.e. this file's own examples/data/.

    Returns
    -------
    built : system_info.BuiltSystem
    score_function : IMP.core.RestraintsScoringFunction
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

    restraints = _add_connectivity_restraints(molecules)
    score_function = IMP.core.RestraintsScoringFunction(restraints)

    built = BuiltSystem(
        model=model,
        system=system,
        state=state,
        root_hier=root_hier,
        dof=dof,
        molecules=molecules,
    )
    return built, score_function
