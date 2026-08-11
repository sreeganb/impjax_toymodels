import os
import string
import unittest


try:
    import IMP
    import IMP.atom
    import IMP.core
    import IMP.pmi.dof
    import IMP.pmi.topology
except ImportError:  # pragma: no cover - environment-dependent
    IMP = None


def _chain_id_alphabet():
    """Yield chain IDs that scale from small to large systems."""
    singles = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    for value in singles:
        yield value
    for first in singles:
        for second in singles:
            yield first + second


def create_molecule(state, name, sequence, chain_id, color):
    """Create one PMI molecule and add a bead-level representation."""
    molecule = state.create_molecule(name, sequence, chain_id=chain_id)
    molecule.add_representation(molecule, resolutions=[1], color=color)
    return molecule


def create_kcoil_ecoil_templates(state, sequences, chain_ids, copy_index):
    """Create one KCOIL/ECOIL pair for a given copy index."""
    k_name = f"KCOIL_copy{copy_index + 1}"
    e_name = f"ECOIL_copy{copy_index + 1}"

    kcoil = create_molecule(
        state=state,
        name=k_name,
        sequence=sequences["K_coil"],
        chain_id=next(chain_ids),
        color="blue",
    )
    ecoil = create_molecule(
        state=state,
        name=e_name,
        sequence=sequences["E_coil"],
        chain_id=next(chain_ids),
        color="red",
    )
    return [kcoil, ecoil]


def create_scaled_system(n_copies):
    """
    Build a scalable KCOIL/ECOIL system using copy-style helper functions.

    Returns:
        (model, system, state, root_hier, dof, molecules)
    """
    if n_copies < 1:
        raise ValueError("n_copies must be >= 1")

    test_dir = os.path.dirname(__file__)
    fasta_file = os.path.join(test_dir, "data", "fasta", "kcoil.fasta")

    model = IMP.Model()
    system = IMP.pmi.topology.System(model, name="Modeling KCOIL-ECOIL multimer")
    state = system.create_state()
    sequences = IMP.pmi.topology.Sequences(fasta_file)

    chain_ids = _chain_id_alphabet()
    molecules = []
    for copy_index in range(n_copies):
        molecules.extend(
            create_kcoil_ecoil_templates(
                state=state,
                sequences=sequences,
                chain_ids=chain_ids,
                copy_index=copy_index,
            )
        )

    root_hier = system.build()
    dof = IMP.pmi.dof.DegreesOfFreedom(model)
    for molecule in molecules:
        dof.create_flexible_beads(molecule)

    return model, system, state, root_hier, dof, molecules


@unittest.skipIf(IMP is None, "IMP/PMI is not installed")
class IMPSystemBuildTests(unittest.TestCase):
    def test_single_copy_builds(self):
        _, _, _, root_hier, dof, molecules = create_scaled_system(n_copies=1)

        self.assertEqual(len(molecules), 2)
        self.assertEqual(molecules[0].get_name(), "KCOIL_copy1")
        self.assertEqual(molecules[1].get_name(), "ECOIL_copy1")
        self.assertGreater(len(IMP.atom.get_leaves(root_hier)), 0)
        self.assertGreater(len(dof.get_movers()), 0)

    def test_copy_scaling_builds_expected_molecule_count(self):
        n_copies = 3
        _, _, _, _, _, molecules = create_scaled_system(n_copies=n_copies)

        self.assertEqual(len(molecules), 2 * n_copies)
        names = {m.get_name() for m in molecules}
        expected = {
            "KCOIL_copy1",
            "ECOIL_copy1",
            "KCOIL_copy2",
            "ECOIL_copy2",
            "KCOIL_copy3",
            "ECOIL_copy3",
        }
        self.assertEqual(names, expected)


if __name__ == "__main__":
    unittest.main()