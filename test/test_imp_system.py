import unittest

try:
    import IMP
    import IMP.atom
    import IMP.core
    import IMP.pmi.dof
    import IMP.pmi.topology
    import IMP.pmi.output
    from IMP.pmi.tools import OrderedSet
    import IMP.pmi.tools
    import IMP.pmi.restraints.stereochemistry
except ImportError:  # pragma: no cover - environment-dependent
    IMP = None

import numpy as np
import os
import itertools
import json

from impjax_toymodels.system_info import BuiltSystem

class BuildIMPSystem:
    def __init__(self, model, system, state):
        """
        A general script used to build any protein complex system in IMP using 
        json files.
        """
        self.model = model
        self.system = system
        self.state = state
#        self.molecules = {}
        self.output_objects = []
        
    def build_component(self, json_file, copy_number = 1):

        mols = []
        with open(json_file) as f:
            protein_info = json.load(f)
        
        repo_dir = os.getcwd()
        
        if protein_info['oligomerization']:
            ch = protein_info['oligomerization_chains']
            factor = copy_number // len(ch)  # Fix for not integer number of copies
            chains = ch * (factor)
        else:
            chains = protein_info['monomer_chain']*copy_number
            print(chains)

        print('Building ...', protein_info['protein_name'])
        print("file path: ", repo_dir, "fasta file: ", protein_info['files']['fasta'])
        seqs = IMP.pmi.topology.Sequences(os.path.join(repo_dir, protein_info['files']['fasta']))
        mol = self.state.create_molecule(protein_info['protein_name'],
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

    def add_connectivity_restraint(self, name = 'Connectivity', molecules = None):
        print("we are here: add_connectivity_restraint")
        print("molecules: ", molecules.items())
        for k, mols in molecules.items():
            print(f'Adding connectivity restraint for molecule group {k}')
            for mol in mols:
                copy_number = IMP.atom.Copy(mol.get_hierarchy()).get_copy_index()
                cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
                cr.set_label(f'{name}.{mol.get_name()}.{copy_number}')
                cr.add_to_model()
                self.output_objects.append(cr)
                print(f'Added connectivity restraint for {mol.get_name()} copy {copy_number}')
        conn_restraint = cr.get_restraint()
        return conn_restraint

    def add_excluded_volume_restraint(self, weight = 1., name = 'ExVol', root_hier = None, molecules = None):
        if not hasattr(self, 'output_objects'):
            self.output_objects = []

        mols = []
        for k, m in molecules.items():
            mols += m
        print('Excluded volume molecules', mols)
        evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=root_hier,
                                                                    resolution=10)
        evr.set_label(f'{name}')
        evr.add_to_model()
        evr.set_weight(weight)
        self.output_objects.append(evr)
        ev_restraint = evr.get_restraint()
        return ev_restraint

    def write_rmf(self, hier, file_out):
        out = IMP.pmi.output.Output()
        out.init_rmf(file_out, [hier])
        out.write_rmf(file_out)
        out.close_rmf(file_out)

    def build_dof(self, json_file, hier, dof, molecules):
        with open(json_file) as f:
            protein_info = json.load(f)
        
        # Get how to build the DOFs
        type_dof = protein_info['degrees_of_freedom']
        name = protein_info['protein_name']
        uniprot_id = protein_info['uniprot_id']
        
        print('###################', name, uniprot_id, type_dof)
        
        if type_dof == 'single':
            for m in molecules[(uniprot_id, name)]:
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
    #repo_dir = "/home/sree/git/impjax_toymodels/test/"
    repo_dir = os.getcwd()
    #repo_dir = "/Users/sreeganeshbalasubramani/git/impjax_toymodels/test/"
    mdl = IMP.Model()
    s = IMP.pmi.topology.System(mdl)
    st = s.create_state()
    molecules = {}
    
    bis = BuildIMPSystem(mdl, s, st)
    info, mols = bis.build_component(repo_dir + "/data/json_files/KCOIL.json", copy_number=4)
    molecules[info] = mols
    
    info, mols = bis.build_component(repo_dir + "/data/json_files/ECOIL.json", copy_number=4)
    molecules[info] = mols
    
    root_hier = s.build()
    
    dof = IMP.pmi.dof.DegreesOfFreedom(mdl)
    dof_built=[]
    for info, mols in molecules.items():
        print("information: ", info[1])
        print("mols", mols)
        if info[1] not in dof_built:
            b = bis.build_dof(repo_dir + f"/data/json_files/{info[1]}.json", root_hier, dof, molecules)
            dof_built.append(b)
        else:
            print(f"Skipped building DOF for {info[1]}")
    
    restraints_set = []
        
    mol_names = []
    for k, ms in molecules.items():
        mol_names += ms
    
    restraints_set.append(bis.add_connectivity_restraint(molecules=molecules))
#    restraints_set.append(bis.add_excluded_volume_restraint(root_hier=root_hier, molecules=molecules))

    sf_imp = IMP.core.RestraintsScoringFunction(restraints_set)
    score = sf_imp.evaluate(False)
    print("Initial score:", score)

    # Separate out the JAX comaptible code here
    # IMP model and scoring functions can be converted to JAX model and JAX scoring functions
    # within IMP
    # Take the sf_imp and convert it to a JAX-compatible scoring function
    ji = sf_imp._get_jax()
    print(ji)
    jax_score_func = ji.score_func 
    jmodel = ji.get_jax_model()
    print("Initial JAX score:", jax_score_func(jmodel))

    
    IMP.pmi.tools.shuffle_configuration(mol_names,
                                    max_translation=300,
                                    avoidcollision_rb=True)
    bis.write_rmf(root_hier, 'system_initial.rmf3')

    print(molecules)

#    sf_imp = IMP.core.RestraintsScoringFunction(restraints_set)
    score = sf_imp.evaluate(False)
    print("post-shuffle IMP score:", score)
    
    built_system = BuiltSystem(
        model=mdl,
        system=s,
        state=st,
        root_hier=root_hier,
        dof=dof,
        molecules=molecules
    )
    built_system.describe(save_file=True)
    
    # Separate out the JAX comaptible code here
    # IMP model and scoring functions can be converted to JAX model and JAX scoring functions
    # within IMP
    # Take the sf_imp and convert it to a JAX-compatible scoring function
    ji = sf_imp._get_jax()
    print(ji)
    jax_score_func = ji.score_func 
    jmodel = ji.get_jax_model()
    print("post-shuffle JAX score:", jax_score_func(jmodel))