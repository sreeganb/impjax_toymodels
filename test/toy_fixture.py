"""Shared toy-system builder for the IMP<->JAX bridge unit tests.

Builds the smallest system that still exercises every code path the bridge
modules (`dof_layout`, `state_sync`, `proposals`, `wrapper_impjax`) care
about: one rigid body (structured KCOIL residues 1-21) plus a run of
flexible beads (the unstructured tail, residues 22-31), scored by a single
connectivity restraint. Not a test itself -- imported by test_*.py files.
"""

import os

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.dof
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.topology

from impjax_toymodels.system_info import BuiltSystem

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_toy_system():
    """Build a one-rigid-body, multi-bead KCOIL system with a connectivity restraint.

    Returns
    -------
    built : BuiltSystem
    restraint_score_function : IMP.core.RestraintsScoringFunction
    """
    model = IMP.Model()
    system = IMP.pmi.topology.System(model)
    state = system.create_state()
    seqs = IMP.pmi.topology.Sequences(os.path.join(DATA_DIR, "fasta", "KCOIL.fasta"))
    mol = state.create_molecule("KCOIL", sequence=seqs["sp"], chain_id="A")
    atomic = mol.add_structure(
        os.path.join(DATA_DIR, "pdb", "KCOIL.pdb"), chain_id="A", res_range=(1, 21)
    )
    mol.add_representation(atomic, resolutions=[1])
    mol.add_representation(mol[:] - atomic, resolutions=[1])
    root_hier = system.build()

    dof = IMP.pmi.dof.DegreesOfFreedom(model)
    selection = IMP.atom.Selection(
        root_hier,
        molecule="KCOIL",
        residue_indexes=range(1, 22),
        resolution=IMP.atom.ALL_RESOLUTIONS,
    ).get_selected_particles()
    _, rigid_body = dof.create_rigid_body(selection, name="KCOIL_0")
    dof.create_flexible_beads(mol.get_non_atomic_residues())

    connectivity = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
    connectivity.add_to_model()

    built = BuiltSystem(
        model=model,
        system=system,
        state=state,
        root_hier=root_hier,
        dof=dof,
        molecules={("K_coil", "KCOIL"): [mol]},
    )
    score_function = IMP.core.RestraintsScoringFunction([connectivity.get_restraint()])
    return built, score_function
