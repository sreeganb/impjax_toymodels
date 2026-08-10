import sys
import IMP
import IMP.core
import IMP.algebra
import IMP.atom
import IMP.container

import IMP.pmi.restraints.crosslinking
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.em
import IMP.pmi.restraints.basic
import IMP.pmi.tools
import IMP.pmi.samplers
import IMP.pmi.output
import IMP.pmi.macros
import IMP.pmi.topology
import ihm.cross_linkers
import jax.numpy as jnp
import string

import sys
import os
from impjax_toymodels.system_info import BuiltSystem

# read in the fasta and pdb 
# Only flexible beads at this point, no rigid bodies 
data_dir = os.path.join(os.getcwd(), "data")
pdb_dir = os.path.join(data_dir, "pdb")
fasta_dir = os.path.join(data_dir, "fasta")

topology_file = os.path.join(data_dir, "topology.txt")

# Declare the number of copies of the molecules that is going to be created
n_copies = 2

def _chain_id_alphabet():
    """A, B, ... Z, 0, ... 9, then AA, AB, ... for large systems"""
    singles = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    for c in singles:
        yield c
    for first in singles:
        for second in singles:
            yield first + second

chain_ids = _chain_id_alphabet()
m = IMP.Model()

# Read in topology file
topology = IMP.pmi.topology.TopologyReader(topology_file,
                                           pdb_dir=pdb_dir,
                                           fasta_dir=fasta_dir)
system = IMP.pmi.topology.System(m, name='Modeling KCOIL-ECOIL multimer')
#st1 = system.create_state()

# Sanity check: print parsed components before building
print("Parsed topology components:")
for c in topology.get_components():
    print(f"  mol={c.molname}  pdb={c.pdb_file}  chain={c.chain}  "
          f"res={c.residue_range}  rb={c.rigid_body}")

bs = IMP.pmi.macros.BuildSystem(m, name="ToyModel", resolutions=[1])
st1 = bs.add_state(topology)
print("What is the state:", st1)
print("print the molecules so you can make copies")
molecules = bs.get_molecules()[0]
print(molecules)
st1.create_copy

root_hier, dof = bs.execute_macro()
output = IMP.pmi.output.Output()
output.init_rmf("ini_all.rmf3", [root_hier])
output.write_rmf("ini_all.rmf3")
output.close_rmf("ini_all.rmf3")

output_objects = [] # keep a list of functions that need to be reported
rmf_restraints = []

print("\nBuild complete.")

# Print molecule/bead counts per type
for state in IMP.atom.get_by_type(root_hier, IMP.atom.STATE_TYPE):
    for mol in IMP.atom.get_by_type(state, IMP.atom.MOLECULE_TYPE):
        leaves = IMP.core.get_leaves(mol)
        print(f"  {mol.get_name()} : {len(leaves)} bead(s)")

molecules = bs.get_molecules()[0]
print("\nMolecules in the system:", molecules)

bsys = BuiltSystem(
    model=m,
    system=system,
    state=st1,
    root_hier=root_hier,
    dof=dof,
    molecules=molecules
)
bsys.describe()

