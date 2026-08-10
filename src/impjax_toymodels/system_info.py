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
    """Everything downstream code needs about a built system."""
    model: "IMP.Model"
    system: "IMP.pmi.topology.System"
    state: "IMP.pmi.topology.State"
    root_hier: "IMP.atom.Hierarchy"
    dof: "IMP.pmi.dof.DegreesOfFreedom"
    molecules: dict = field(default_factory=dict)   # (name, copy_index) -> Molecule
    spec: Optional["IMP.pmi.topology.Specification"] = None

    @property
    def n_copies(self):
        return self.spec.n_copies if self.spec else 1

    @property
    def molecule_names(self):
        return [m.name for m in self.spec.molecules] if self.spec else []

    def copies_of(self, molecule_name):
        """PMI Molecule objects for every copy of `molecule_name`, in order."""
        return [self.molecules[(molecule_name, i)] for i in range(self.n_copies)]

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
        print(f"Built system '{self.spec.name if self.spec else '?'}' "
              f"with {self.n_copies} copy/copies")
        for (name, copy_index), mol in sorted(self.molecules.items()):
            n_leaves = len(IMP.core.get_leaves(mol.get_hierarchy()))
            chain = mol.get_hierarchy().get_name()
            print(f"  {name}.{copy_index:<3d} chain={chain:<4s} beads={n_leaves}")
        print(f"  rigid bodies  : {len(rigid_bodies)}")
        print(f"  flexible beads: {len(beads)}")
        print(f"  movers        : {len(self.dof.get_movers())}")
