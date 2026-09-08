"""Build the 1AVX trypsin/inhibitor docking system.

A system module in the sense of examples/harness/system_registry.py: it is the
only file in this example that knows what the molecules are.  The harness
generates its contact map, selects its restraints, runs every sampler on it
and measures the result without naming trypsin once.

The system
----------
Two molecules, one rigid body each -- TRYP (223 residues) and STI (172) --
built from the renumbered chains prepare_1avx.py writes.  With gaps spliced
out every residue is structured, so PMI creates no flexible beads and the
whole sampling problem is 2 x (quaternion + translation): 14 parameters, 12
degrees of freedom, of which only the 6 relative ones change the score.

That is deliberately different from the KCOIL/ECOIL example, which is mostly a
question about a flexible linker.  Here nothing connects the two bodies: the
only thing pulling them together is the synthetic interface restraints, which
makes this a genuine docking search rather than a refinement.

Restraint arrangement
---------------------
`build_split` puts the harmonic interface restraints -- the stand-in for
crosslinking data -- on the tempered likelihood side, and excluded volume on
the untempered prior side.  Excluded volume is structural prior knowledge
about proteins, not an experimental observation, so this is the more
defensible partition; it also gives SMC something to do, because the
distance-restraint term varies enormously across draws from a box prior,
which is exactly the discriminating power the tempering schedule needs.  (In
the KCOIL/ECOIL split it was excluded volume that was tempered, and its
variance across prior draws is nearly zero -- box-scattered particles never
clash -- so the anneal there was close to a no-op.)

Connectivity is deliberately absent unless the representation actually has
flexible beads.  With one rigid body per chain, every consecutive pair of
beads sits in the same rigid body, so the connectivity restraint evaluates to
a constant: it cannot distinguish two configurations, and every restraint is
a node in the exported JAX graph, so scoring one is pure cost.  Build with
`prepare_1avx.py --model-gaps` and it comes back automatically.

Measured: the JAX excluded-volume score carries a constant offset
----------------------------------------------------------------
IMP has no JAX path for close-pair containers and says so
("No JAX support for close pairs, so using all possible pairs").  On the CPU,
`ExcludedVolumeSphere` filters out pairs of beads that share a rigid body; the
all-pairs JAX export cannot, and consecutive coarse beads along a chain
overlap by construction (a resolution-2 bead has radius ~3-4 A and its
neighbour's centre is ~5 A away).  On this system, at the crystal pose:

    all-pairs total   466.1924    <- what score_func returns
      intra-body      455.6515    <- 374 overlapping pairs, both ends in one body
      inter-body       10.5408    <- 7 pairs; equals IMP's CPU score exactly

The intra-body part is an **exact additive constant** in the sampled
posterior, because those bead pairs are rigid-body members and their mutual
distances cannot change: measured at 455.6515 +/- 2e-4 (float32 rounding in
the quaternion rotation) across rigid-body moves that swing the inter-body
term from 0 to 56.6.  A constant in the log-density cancels in every
Metropolis acceptance ratio, and -- since excluded volume sits in the
untempered prior here -- in SMC's incremental weights too.  So sampling is
unaffected; only the absolute number differs from IMP's, which is why the
benchmark re-scores frames with IMP on the CPU rather than quoting the
BlackJAX log-posterior.  Expect DEBUG score verification to report this
offset; it is understood, not a bug in this repo's expansion map.
"""

import json
import os
from typing import Dict, List, Tuple

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.dof
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.tools
import IMP.pmi.topology

from impjax_toymodels import distance_restraints
from impjax_toymodels.system_info import BuiltSystem

#: Base directory whose "data" subfolder holds this system's inputs.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

#: Molecule names, in the fixed order every bead ordering downstream follows.
#: TRYP first makes it the "receptor" for the CAPRI-style ligand RMSD, which
#: is the convention for the larger partner.
PROTEINS = ("TRYP", "STI")

#: Default restraint file, relative to DATA_DIR.
DEFAULT_DISTANCE_CSV = os.path.join("data", "distance_constraints.csv")

#: Excluded-volume selection resolution.  IMP has no JAX path for close-pair
#: containers ("No JAX support for close pairs, so using all possible pairs"),
#: so this term costs O(n^2) in the exported graph and the bead count it acts
#: on is the single biggest lever on compile and evaluation time here.
EXCLUDED_VOLUME_RESOLUTION = 10

#: How far shuffle_configuration may throw a rigid body from where it was
#: built.  Much smaller than the KCOIL/ECOIL example's 200 A: these bodies are
#: only ~40 A across and, with no connectivity tethering them, a start that is
#: hundreds of angstroms away leaves the harmonic wells as the sole restoring
#: force over that whole distance.
DEFAULT_SHUFFLE_TRANSLATION = 50.0

MoleculeKey = Tuple[str, str]
MoleculesByKey = Dict[MoleculeKey, List]


def _load(json_path: str) -> dict:
    with open(json_path) as handle:
        return json.load(handle)


def _json_path(protein: str, data_dir: str = None) -> str:
    return os.path.join(data_dir or DATA_DIR, "data", "json_files", f"{protein}.json")


def domains_for(protein: str, data_dir: str = None) -> List[Tuple[int, int]]:
    """Residue ranges that become rigid bodies, from the protein's JSON.

    prepare_1avx.py derives these from which residues are actually observed,
    so they are one range per chain by default and two for STI when the
    unobserved 639-643 loop is modelled.
    """
    info = _load(_json_path(protein, data_dir))
    return [(int(low), int(high)) for low, high in info["domains"]]


def _build_component(state, protein: str, data_dir: str, copy_number: int):
    """Create `copy_number` copies of one protein from its JSON description."""
    info = _load(_json_path(protein, data_dir))
    chain = info["monomer_chain"][0]

    sequences = IMP.pmi.topology.Sequences(
        os.path.join(data_dir, info["files"]["fasta"]))
    pdb_path = os.path.join(data_dir, info["files"]["pdb"])

    root_mol = state.create_molecule(
        info["protein_name"], sequence=sequences["sp"], chain_id=chain)

    mols = []
    for copy_index in range(copy_number):
        target = root_mol if copy_index == 0 else root_mol.create_copy(chain_id=chain)
        structured = IMP.pmi.tools.OrderedSet()
        for low, high in info["domains"]:
            residues = target.add_structure(
                pdb_path, chain_id=chain, res_range=(low, high))
            target.add_representation(
                residues,
                resolutions=info["representation"]["structured"],
                color=info["visualization"]["color"])
            for particle in residues:
                structured.add(particle)
        # Whatever is left is unobserved in the crystal.  With the default
        # preparation there is nothing left and this adds no beads at all;
        # with --model-gaps it picks up the 5-residue STI loop.
        remainder = target[:] - structured
        if remainder:
            target.add_representation(
                remainder,
                resolutions=info["representation"]["unstructured"],
                color=info["visualization"]["color"])
        mols.append(target)

    return (info["uniprot_id"], info["protein_name"]), mols


def _build_rigid_bodies_and_flexible_beads(dof, root_hier, protein: str,
                                           data_dir: str, molecules: MoleculesByKey) -> None:
    """One rigid body per structured domain; beads for anything unobserved."""
    info = _load(_json_path(protein, data_dir))
    key = (info["uniprot_id"], info["protein_name"])
    name = info["protein_name"]

    for mol in molecules[key]:
        copy_index = IMP.atom.Copy(mol.get_hierarchy()).get_copy_index()
        for low, high in info["domains"]:
            selection = IMP.atom.Selection(
                root_hier,
                molecule=name,
                residue_indexes=range(low, high + 1),
                copy_index=copy_index,
                resolution=IMP.atom.ALL_RESOLUTIONS,
            ).get_selected_particles()
            dof.create_rigid_body(selection, name=f"{name}_{copy_index}_{low}-{high}")
        dof.create_flexible_beads(mol.get_non_atomic_residues())


def _add_connectivity_restraints(molecules: MoleculesByKey) -> list:
    """Connectivity, but only for molecules that actually have flexible beads.

    A molecule represented as a single rigid body has every consecutive bead
    pair at a fixed distance, so its connectivity restraint is a constant: it
    cannot distinguish two configurations, yet it still becomes a node in the
    exported JAX graph and is evaluated at every step.  Adding it would be
    pure cost.  See the module docstring.
    """
    restraints = []
    for mols in molecules.values():
        for mol in mols:
            if not mol.get_non_atomic_residues():
                continue
            copy_index = IMP.atom.Copy(mol.get_hierarchy()).get_copy_index()
            connectivity = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
            connectivity.set_label(f"Connectivity.{mol.get_name()}.{copy_index}")
            connectivity.add_to_model()
            restraints.append(connectivity)
    return restraints


def _add_excluded_volume_restraint(root_hier, resolution: int):
    """One excluded-volume term over everything, keeping the bodies apart."""
    return IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
        included_objects=root_hier, resolution=resolution)


def _add_distance_restraints(root_hier, copy_number: int, csv_path: str) -> list:
    """Attach every harmonic restraint listed in a constraint CSV.

    The restraints are data, not code: which residue pairs, at what distance,
    with what force constant all live in the CSV that
    harness/generate_distance_restraints.py selects out of the ground-truth
    contact map.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"distance constraint file not found: {csv_path}\n"
            "Generate it from the ground-truth structure first:\n"
            "    python harness/generate_distance_restraints.py --system docking_system ...")
    return distance_restraints.load_and_add(root_hier, csv_path, copy_number)


def _build_system_and_restraints(copy_number: int, data_dir: str, shuffle: bool,
                                 distance_csv, shuffle_translation: float,
                                 excluded_volume_resolution: int):
    """Shared construction, so the combined and split arrangements agree.

    Parameters
    ----------
    copy_number : copies of *each* protein.  Docking is a 1:1 complex, so this
        is 1 in every normal use; it exists so the harness can put a second
        point on its scaling page and so the module satisfies the protocol.
    shuffle : randomize the starting configuration.  True is what sampling
        wants -- inference has to start away from the answer.  False leaves
        both rigid bodies on their crystal coordinates, which is the ground
        truth the restraints were measured off and the reference every model
        is scored against.
    distance_csv : restraint file path, or False to build with none (the
        bootstrap path, used while *writing* the restraint file).
    """
    base_dir = data_dir or DATA_DIR
    model = IMP.Model()
    system = IMP.pmi.topology.System(model)
    state = system.create_state()

    molecules: MoleculesByKey = {}
    for protein in PROTEINS:
        key, mols = _build_component(state, protein, base_dir, copy_number)
        molecules[key] = mols

    root_hier = system.build()
    dof = IMP.pmi.dof.DegreesOfFreedom(model)
    for protein in PROTEINS:
        _build_rigid_bodies_and_flexible_beads(dof, root_hier, protein, base_dir, molecules)

    connectivity = _add_connectivity_restraints(molecules)
    excluded_volume = _add_excluded_volume_restraint(root_hier, excluded_volume_resolution)
    if distance_csv is False:
        disres = []
    else:
        csv_path = distance_csv or os.path.join(base_dir, DEFAULT_DISTANCE_CSV)
        disres = _add_distance_restraints(root_hier, copy_number, csv_path)

    if shuffle:
        every_molecule = [mol for mols in molecules.values() for mol in mols]
        IMP.pmi.tools.shuffle_configuration(
            every_molecule,
            max_translation=shuffle_translation,
            avoidcollision_rb=True)

    built = BuiltSystem(
        model=model,
        system=system,
        state=state,
        root_hier=root_hier,
        dof=dof,
        molecules=molecules,
    )
    return built, connectivity, excluded_volume, disres


def build_system(copy_number: int = 1, data_dir: str = None, shuffle: bool = True,
                 distance_csv=None,
                 shuffle_translation: float = DEFAULT_SHUFFLE_TRANSLATION,
                 excluded_volume_resolution: int = EXCLUDED_VOLUME_RESOLUTION):
    """Build the docking system with every restraint in one scoring function.

    This is the arrangement IMP's own replica exchange is run on, so that the
    BlackJAX and IMP baselines score exactly the same target.

    Returns
    -------
    built : system_info.BuiltSystem
    score_function : IMP.core.RestraintsScoringFunction over all restraints
    output_objects : PMI restraint objects, for IMP's stat-file machinery
    """
    built, connectivity, excluded_volume, disres = _build_system_and_restraints(
        copy_number, data_dir, shuffle, distance_csv,
        shuffle_translation, excluded_volume_resolution)

    output_objects = list(connectivity) + [excluded_volume] + disres
    restraints = ([r.get_restraint() for r in connectivity]
                  + [excluded_volume.get_restraint()]
                  + [r.get_restraint() for r in disres])
    return built, IMP.core.RestraintsScoringFunction(restraints), output_objects


def build_split(copy_number: int = 1, data_dir: str = None, shuffle: bool = True,
                distance_csv=None,
                shuffle_translation: float = DEFAULT_SHUFFLE_TRANSLATION,
                excluded_volume_resolution: int = EXCLUDED_VOLUME_RESOLUTION):
    """Build the same system with its restraints partitioned prior/likelihood.

    Likelihood (tempered): the harmonic interface restraints -- the synthetic
    crosslinking data.  Prior (untempered): excluded volume, plus connectivity
    if the representation has flexible beads at all.  See the module docstring
    for why this partition rather than the KCOIL/ECOIL one.

    The two scoring functions are disjoint by construction, which is what
    priors.restraint_prior requires: a restraint in both would be counted
    twice at lambda = 1 and quietly change the posterior.

    Returns
    -------
    built : system_info.BuiltSystem
    likelihood_score_function : the distance restraints
    prior_score_function : excluded volume (+ connectivity if present)
    output_objects : PMI restraint objects, for IMP's stat-file machinery
    """
    built, connectivity, excluded_volume, disres = _build_system_and_restraints(
        copy_number, data_dir, shuffle, distance_csv,
        shuffle_translation, excluded_volume_resolution)

    likelihood_sf = IMP.core.RestraintsScoringFunction(
        [r.get_restraint() for r in disres])
    prior_sf = IMP.core.RestraintsScoringFunction(
        [excluded_volume.get_restraint()] + [r.get_restraint() for r in connectivity])
    return built, likelihood_sf, prior_sf, list(connectivity) + [excluded_volume] + disres
