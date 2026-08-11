import unittest

try:
    import IMP
    import IMP.atom
    import IMP.core
    import IMP.pmi.dof
    import IMP.pmi.topology
    from IMP.pmi.tools import OrderedSet
except ImportError:  # pragma: no cover - environment-dependent
    IMP = None

import numpy as np
import os
import itertools
import json

class BuildIMPSystem:
    def __init__(self, model, system, state):
        """
        A general script used to build any protein complex system in IMP using 
        json files.
        """
        self.model = model
        self.system = system
        self.state = state
        self.molecules = {}
        
    def build_component(self, json_file, copy_number = 1):

        mols = []
        with open(json_file) as f:
            protein_info = json.load(f)

        
        if protein_info['oligomerization']:
            ch = protein_info['oligomerization_chains']
            factor = copy_number // len(ch)  # Fix for not integer number of copies
            chains = ch * (factor)
        else:
            chains = protein_info['monomer_chain']*copy_number
            print(chains)

        print('Building ...', protein_info['protein_name'])
        seqs = IMP.pmi.topology.Sequences(os.path.join(repo_dir,protein_info['files']['fasta']))
        mol = st.create_molecule(protein_info['protein_name'],
                                sequence = seqs['sp'],
                                chain_id = chains[0])
        
        # Add all domains
        domains = protein_info['domains']
        all_atomic = OrderedSet()
        for d in domains:
            print('Copy 0 ...')
            atomic = mol.add_structure(os.path.join(repo_dir,protein_info['files']['pdb']),
                                    chain_id=chains[0],
                                    res_range =d)
            mol.add_representation(atomic,                
                                resolutions=protein_info['representation']['structured'],
                                color=protein_info['visualization']['color'])
            for a in atomic:
                all_atomic.add(a)

        # Add flexible linkers 
        mol.add_representation(mol[:]-all_atomic,                
                            resolutions=protein_info['representation']['unstructured'],
                            color=protein_info['visualization']['color'])
        mols.append(mol)

        if copy_number > 1:
            all_atomic = OrderedSet()
            for i in np.arange(1,copy_number,1):
                print(f'Copy {i} ...')
                copy =  mol.create_copy(chain_id = chains[i])
                all_atomic = OrderedSet()
                for d in domains:
                    atomic = copy.add_structure(os.path.join(repo_dir,protein_info['files']['pdb']),
                                                chain_id=chains[i],
                                                res_range =d)

                    copy.add_representation(atomic,                
                                            resolutions=protein_info['representation']['structured'],
                                            color=protein_info['visualization']['color'])
                
                    for a in atomic:
                        all_atomic.add(a)

                # Add flexible linkers
                copy.add_representation(copy[:]-all_atomic,                
                                        resolutions=protein_info['representation']['unstructured'],
                                        color=protein_info['visualization']['color'])

                mols.append(copy)

        return (protein_info["uniprot_id"],protein_info['protein_name']), mols

    def write_rmf(self, hier, file_out):
        out = IMP.pmi.output.Output()
        out.init_rmf(file_out, [hier])
        out.write_rmf(file_out)
        out.close_rmf(file_out)

    def build_dof(self, json_file, molecules):
        with open(json_file) as f:
            protein_info = json.load(f)
        
        # Get how to build the DOFs
        type_dof = protein_info['degrees_of_freedom']
        name = protein_info['protein_name']
        uniprot_id = protein_info['uniprot_id']
        
        print('###################', name, uniprot_id, type_dof)
        
        if type_dof == 'single':
            for m in molecules(uniprot_id, name):
                copy = IMP.atom.Copy(m.get_hierarchy()).get_copy_index()
                sel =  IMP.atom.Selection(hier,
                                        molecule=name,
                                        resolution=IMP.atom.ALL_RESOLUTIONS).get_selected_particles()
                rb_movers,rb = dof.create_rigid_body(sel,
                                                    name = f'{name}_{copy}')
                dof.create_flexible_beads(m.get_non_atomic_residues())

            return [protein_info['uniprot_id']]
                
        elif type_dof == 'domains':
            domains = protein_info['domains']
            print('Molecules', molecules[(uniprot_id, name)])
            for mol in molecules[(uniprot_id, name)]:
                copy = IMP.atom.Copy(mol.get_hierarchy()).get_copy_index()
                for d in domains:
                    sel =  IMP.atom.Selection(hier,
                                            molecule=name,
                                            residue_indexes=range(d[0],d[1]+1,1),
                                            copy_index=copy,
                                            resolution=IMP.atom.ALL_RESOLUTIONS).get_selected_particles()
                    rb_movers,rb = dof.create_rigid_body(sel,
                                                        name = f'{name}_{copy}')
                dof.create_flexible_beads(mol.get_non_atomic_residues())
            return [protein_info['uniprot_id']]

        elif type_dof == 'custom':
            print('Custom DOF for', name)
            definition = protein_info["degrees_of_freedom_definition"]
            print('!!!!!!!!!!!! definition', definition)

            
            copy_number = len(molecules[(uniprot_id, name)])
            def_values = definition.values()
            ch = 0
            for val in def_values:
                ch += np.sum([1 for tup in val if tup[2] == name])

            factor = copy_number // ch
            print('factor', factor, def_values, ch)
            if factor == 0: ### Fix this
                factor = 1


            prots = []
            for m in np.arange(0, 2*factor , 2): # Fix this
                print('-------', name, m)
                for k, rigid_body in definition.items():
                    sel = []
                    for c in rigid_body:
                        sel +=  IMP.atom.Selection(hier,
                                                molecule=c[2],
                                                residue_indexes=range(c[0],c[1]+1,1),
                                                
    copy_index=c[3] + m,
                                                resolution=IMP.atom.ALL_RESOLUTIONS).get_selected_particles()
                        prots.append(c[2])
                    rb_movers,rbs = dof.create_rigid_body(sel,
                                                        name = f'{k}_1')
                    
                        

            # Now select all proteins to create flexible beads
            set_proteins = list(set(prots))
            selected_molecules = []
            for p in set_proteins:
                for k, v in molecules.items():
                    print('k',p, k)
                    print('v', v)
                selected_molecules.append([v for k,v in molecules.items() if k[1]==p][0])
                
            selected_molecules_flat = list(itertools.chain.from_iterable(selected_molecules))
            print(selected_molecules)
            # Flexible mover for all molecules 
            for mol in selected_molecules_flat:
                print('mol', mol)
                flex_movers = dof.create_flexible_beads(mol.get_non_atomic_residues())
            
            return set_proteins

        else:
            return ['None']
#@unittest.skipIf(IMP is None, "IMP/PMI is not installed")
#class IMPSystemBuildTests(unittest.TestCase):

if __name__ == "__main__":
    repo_dir = "/home/sree/git/impjax_toymodels/test/"
    
    mdl = IMP.Model()
    s = IMP.pmi.topology.System(mdl)
    st = s.create_state()
    molecules = {}
    
    bis = BuildIMPSystem(mdl, s, st)
    info, mols = bis.build_component(repo_dir + "data/json_files/KCOIL.json", copy_number=2)
    molecules[info] = mols
    
    info, mols = bis.build_component(repo_dir + "data/json_files/ECOIL.json", copy_number=2)
    molecules[info] = mols
    
    root_hier = s.build()
    bis.write_rmf(root_hier, 'system_initial.rmf3')

    print(molecules)
    
    dof = IMP.pmi.dof.DegreesOfFreedom(mdl)
    

    #unittest.main()
