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
import IMP.pmi.tools
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


def build_split_toy_system(shuffle_translation: float = 50.0):
    """The same toy system, but with its restraints partitioned into a prior
    set and a likelihood set -- the arrangement priors.restraint_prior exists
    to serve (connectivity as the structural prior, excluded volume standing
    in for a data-derived likelihood term).

    The configuration is shuffled first so both terms actually vary: as
    built, the connectivity restraint sits at exactly 0 and would make every
    tempering step a no-op.

    Returns
    -------
    built : BuiltSystem
    likelihood_score_function : excluded volume only
    prior_score_function : connectivity only
    """
    built, connectivity_sf = build_toy_system()
    IMP.pmi.tools.shuffle_configuration(built.root_hier, max_translation=shuffle_translation)

    excluded_volume = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
        included_objects=built.root_hier, resolution=1
    )
    excluded_volume.add_to_model()
    likelihood_sf = IMP.core.RestraintsScoringFunction([excluded_volume.get_restraint()])

    # Both must be evaluated once so IMP's JAX export is materialized.
    likelihood_sf.evaluate(False)
    connectivity_sf.evaluate(False)
    return built, likelihood_sf, connectivity_sf


def build_rigid_only_system():
    """Two rigid bodies and *no* flexible beads -- the docking-shaped system.

    Every bridge module carries a bead branch (`bead_coords` in theta, a bead
    proposal, a bead row-set in the expansion map Phi), and until now every
    fixture had beads for it to act on.  A rigid-body docking system has none,
    so those branches all run at size zero: empty index arrays, (0, 3) arrays
    and empty `.at[].set()` updates.  This fixture exists to hold that path
    open, since an empty-array shape or dtype slip there fails only for
    systems with no beads and would otherwise go unnoticed.

    Two copies of the structured KCOIL domain (residues 1-21), each its own
    rigid body, so the excluded-volume restraint has something to score and
    the rotation/translation proposals have more than one body to act on.
    Only the structured representation is added, so `get_non_atomic_residues`
    contributes nothing and PMI creates no beads.

    Returns
    -------
    built : BuiltSystem
    score_function : IMP.core.RestraintsScoringFunction (excluded volume)
    """
    model = IMP.Model()
    system = IMP.pmi.topology.System(model)
    state = system.create_state()
    seqs = IMP.pmi.topology.Sequences(os.path.join(DATA_DIR, "fasta", "KCOIL.fasta"))

    mol = state.create_molecule("KCOIL", sequence=seqs["sp"], chain_id="A")
    molecules = {}
    for copy_index in range(2):
        target = mol if copy_index == 0 else mol.create_copy(chain_id="B")
        atomic = target.add_structure(
            os.path.join(DATA_DIR, "pdb", "KCOIL.pdb"), chain_id="A", res_range=(1, 21)
        )
        # Structured representation only: no add_representation over the
        # unstructured remainder, hence no flexible beads anywhere.
        target.add_representation(atomic, resolutions=[1])
        molecules[("K_coil", "KCOIL", copy_index)] = [target]
    root_hier = system.build()

    dof = IMP.pmi.dof.DegreesOfFreedom(model)
    for copy_index in range(2):
        selection = IMP.atom.Selection(
            root_hier,
            molecule="KCOIL",
            copy_index=copy_index,
            residue_indexes=range(1, 22),
            resolution=IMP.atom.ALL_RESOLUTIONS,
        ).get_selected_particles()
        dof.create_rigid_body(selection, name=f"KCOIL_{copy_index}")

    excluded_volume = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
        included_objects=root_hier, resolution=1
    )
    excluded_volume.add_to_model()

    built = BuiltSystem(
        model=model,
        system=system,
        state=state,
        root_hier=root_hier,
        dof=dof,
        molecules={("K_coil", "KCOIL"): [mol]},
    )
    score_function = IMP.core.RestraintsScoringFunction([excluded_volume.get_restraint()])
    score_function.evaluate(False)  # materialize IMP's JAX export
    return built, score_function
