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

    def describe(self, save_file=False):
        """Print a compact inventory of what was actually built."""
        rigid_bodies, beads = self.rigid_bodies_and_beads()
        print(f"Built system with {len(self.molecule_names)} molecule types")
        for (name, copy_index), mol in sorted(self.molecules.items()):
            n_leaves = len(IMP.core.get_leaves(self.root_hier))
            chain = self.root_hier.get_name()
            print(f"  {name}.{copy_index} chain={chain} beads={n_leaves}")
        print(f"  rigid bodies  : {len(rigid_bodies)}")
        print(f"  flexible beads: {len(beads)}")
        print(f"  movers        : {len(self.dof.get_movers())}")
        # If save_file = true, then save the quaternions of rigid bodies and coordinates of 
        # flexible beads 
        if save_file:
            with open("rigid_bodies_and_beads.txt", "w") as f:
                f.write("Rigid bodies:\n")
                for rb in rigid_bodies:
                    f.write(f"{rb}\n")
                f.write("Flexible beads:\n")
                for bead in beads:
                    f.write(f"{bead}\n")

    def read_in_jax_model(self, jax_model):
        """
        Update the IMP system with a JAX model.

        The JAX model is assumed to be in the same order as the rigid bodies and
        beads returned by `rigid_bodies_and_beads()`.
        """
        rigid_bodies, beads = self.rigid_bodies_and_beads()
        assert len(jax_model) == len(rigid_bodies) + len(beads)
        for i, rb in enumerate(rigid_bodies):
            q = jax_model[i]
            IMP.core.RigidMember(rb).set_quaternion(q)
        for i, bead in enumerate(beads):
            pos = jax_model[len(rigid_bodies) + i]
            IMP.core.XYZ(bead).set_coordinates(pos)