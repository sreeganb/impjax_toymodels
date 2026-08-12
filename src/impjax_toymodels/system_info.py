import os
import string
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.dof
import IMP.pmi.topology

@dataclass
class BuiltSystem:
    def __init__(self, model, system, state, root_hier, dof, molecules):
        self.model = model
        self.system = system
        self.state = state
        self.root_hier = root_hier
        self.dof = dof
        self.molecules = molecules
   
    @property
    def molecule_names(self):
        return list({name for (name, copy_index) in self.molecules.keys()})

    def copies_of(self, molecule_name):
        """PMI Molecule objects for every copy of `molecule_name`, in order."""
        return [self.molecules[(molecule_name, i)] for i in range(self.copies_of_molecule(molecule_name))]

    def rigid_bodies_and_beads(self):
        """
        Split all leaves into (ordered rigid bodies, flexible/non-rigid beads).

        First-appearance order is preserved so indices stay stable across
        calls -- the JAX adapter relies on this.
        """
        seen = set()
        rigid_bodies = []
        beads = []
        for particle in IMP.atom.get_leaves(self.root_hier):
            rb = None
            if IMP.core.RigidMember.get_is_setup(particle):
                rb = IMP.core.RigidMember(particle).get_rigid_body()
            elif IMP.core.NonRigidMember.get_is_setup(particle):
                rb = IMP.core.NonRigidMember(particle).get_rigid_body()
                beads.append(particle)
            else:
                beads.append(particle)
            if rb is not None and rb not in seen:
                seen.add(rb)
                rigid_bodies.append(rb)
        return rigid_bodies, beads

    def describe(self):
        """Print a compact inventory of what was actually built."""
        rigid_bodies, beads = self.rigid_bodies_and_beads()
        print(f"Built system with {len(self.molecule_names)} molecule types")
        for (name, copy_index), mol in sorted(self.molecules.items()):
            n_leaves = len(IMP.core.get_leaves(self.root_hier))
            chain = self.root_hier.get_name()
            print(f"  {name}.{copy_index:<3d} chain={chain:<4s} beads={n_leaves}")
        print(f"  rigid bodies  : {len(rigid_bodies)}")
        print(f"  flexible beads: {len(beads)}")
        print(f"  movers        : {len(self.dof.get_movers())}")
