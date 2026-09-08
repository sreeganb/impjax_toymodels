"""Sweep copy numbers and samplers, then report timings, scores and accuracy.

A standalone harness over everything else in examples/: it builds no systems
and runs no samplers of its own, it drives run_sampling_comparison.py's
runners and generate_distance_restraints.py's selection, then measures what
came out and writes a single multi-page PDF.  Which system is swept comes
from the config's "system" key, so this file names no molecules.

    python harness/benchmark_pipeline.py --config data/benchmark_config.json

What it measures, and why each is separate
------------------------------------------
* **Wall and CPU time** per (copy number, sampler), with the JAX backend
  recorded, since the same config on CPU and on GPU is the comparison that
  matters and only the machine can say which one ran.
* **RMSD to the ground truth**, over a window of final frames, superposed and
  over structured beads only. This is the accuracy question.
* **The IMP score of those same frames**, re-evaluated on the CPU by loading
  each frame back into an IMP model. Deliberately *not* the BlackJAX
  log-posterior: that is `-S(theta) + log p0(theta)`, a different number, and
  quoting it would make BlackJAX and IMP samplers incomparable. Re-scoring is
  cheap (~0.6 ms/frame), so the window costs nothing.
* **Restraint satisfaction** -- how many restraints are within a tolerance in
  those frames. More diagnostic than an aggregate score, which can hide two
  badly violated restraints under a hundred satisfied ones.
* **Scaling** -- system size (sampled degrees of freedom) against wall time.
* **Whatever a config's `metrics_module` adds** -- for the docking example,
  the CAPRI ligand/interface RMSDs and fraction of native contacts.

Every case gets a freshly generated restraint set from the contact map for
its copy number, so restraint count is a controlled variable rather than an
accident of what was lying in data/.

Seeds
-----
A config's "seeds" list runs the whole sweep once per seed, and each seed is
a different starting structure (run_sampling_comparison.seed_imp seeds IMP's
own RNG, which is what `shuffle_configuration` draws from). On a multimodal
target one seed measures luck as much as sampler quality: the same config was
measured docking to 1.6 A on one run of this example and failing at 17.6 A on
the next, before seeding was fixed.
"""

import argparse
import copy as copy_module
import importlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import IMP
import IMP.atom
import IMP.core
import IMP.pmi.restraints.basic
import IMP.rmf
import RMF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate_recovery
import generate_distance_restraints
import run_sampling_comparison as runner
import system_registry
from impjax_toymodels import contact_map, dof_layout, distance_restraints, logging_config

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Filled in from the JSON, but every key needs a value before a Namespace can
#: be handed to run_sampling_comparison's runners.
SAMPLER_DEFAULTS = {
    "n_steps": 5000, "burnin": 1000, "thin": 1,
    "n_particles": 100, "n_temperature_steps": 100, "n_mcmc_steps": 10,
    "schedule": "linear", "target_ess": 0.5,
    "imp_rex_frames": 5000, "imp_rex_mc_steps": 10, "imp_rex_max_temp": 2.5,
}

PROPOSAL_DEFAULTS = {
    "mode": "all", "sigma_rotation": 0.05,
    "sigma_translation": 1.0, "sigma_bead": 1.0,
}


def load_config(path: str) -> dict:
    """Read a sweep config and attach the system module it names.

    Every relative path in the config -- ground-truth structures, contact
    maps, the output directory -- resolves against that system's own base
    directory, so a config lives next to the data it points at rather than
    being written relative to examples/.
    """
    with open(path) as handle:
        config = json.load(handle)
    for required in ("copy_numbers", "samplers", "ground_truth"):
        if required not in config:
            raise ValueError(f"{path}: missing required key {required!r}")
    config.setdefault("system", system_registry.DEFAULT_SYSTEM)
    # Cached under a private key so it is never written back out to results.json.
    config["_system"] = system_registry.resolve(config["system"])
    config["_base_dir"] = config["_system"].DATA_DIR
    # Optional per-system extra metrics (e.g. the CAPRI docking scores).  The
    # module must expose build_evaluator(system) -> (evaluator, reference) and
    # an evaluator with .score(coords) -> object with .as_row().
    config["_metrics"] = (
        importlib.import_module(config["metrics_module"])
        if config.get("metrics_module") else None)
    config.setdefault("name", "benchmark")
    config.setdefault("output_dir", "out/benchmark")
    config.setdefault("score_window", 50)
    # Cap the window at the last quarter of a run, so a short trajectory (an
    # SMC ladder writes one frame per temperature step) is not judged on its
    # own anneal. See scoring_window.
    config.setdefault("score_window_fraction", 0.25)
    config.setdefault("seed", 0)
    # A list of seeds runs the whole sweep once per seed. Each seed is a
    # different starting structure (see run_sampling_comparison.seed_imp), so
    # this is what turns a single anecdote into a distribution.
    config.setdefault("seeds", [config["seed"]])
    config.setdefault("prior", "connectivity+box")
    config.setdefault("satisfaction_tolerance", 2.0)
    config.setdefault("restraints", {})
    config.setdefault("proposal", {})
    config.setdefault("sampler_params", {})
    return config


def ground_truth_for(config: dict, copy_number: int) -> dict:
    """Resolve which structure and contact map describe this copy number.

    `per_copy_number` lets a real N-copy prediction override the default, and
    is the path the AF3-per-copy-number workflow takes. Without an override
    the single reference assembly is replicated, which is what
    `wildcard_copies` in the restraint block expresses.
    """
    spec = dict(config["ground_truth"])
    override = spec.pop("per_copy_number", {}) or {}
    spec.update(override.get(str(copy_number), {}))
    return spec


def chain_args(chains: dict) -> List[str]:
    """Render a {chain: [protein, copy]} mapping as --chains arguments."""
    return [f"{chain}={protein}:{index}" for chain, (protein, index) in chains.items()]


def prepare_restraints(config: dict, copy_number: int, case_dir: str) -> str:
    """Generate this copy number's restraint file from its contact map."""
    truth = ground_truth_for(config, copy_number)
    options = {**{"top_n": 10, "top_n_intra": 2, "residue_types": "KS",
                  "min_seq_sep": 3, "force_constant": 1.0, "wildcard_copies": True},
               **config["restraints"]}
    output = os.path.join(case_dir, "distance_constraints.csv")
    contact_map_path = os.path.join(config["_base_dir"], truth["contact_map"])

    # A '*' copy wildcard can only say "copy i to copy i", so it is valid only
    # when the ground truth describes a single assembly that gets replicated.
    # A genuine multi-copy structure has real cross-copy contacts, and those
    # have to be written with explicit copy indexes.
    copies_in_map = {c for pair in contact_map.read_contact_map(contact_map_path)
                     for c in (pair.copy1, pair.copy2)}
    if len(copies_in_map) > 1 and options["wildcard_copies"]:
        print(f"  contact map spans copies {sorted(copies_in_map)}; writing explicit "
              "copy indexes rather than the '*' wildcard")
        options["wildcard_copies"] = False

    argv = [
        "--contact-map", contact_map_path,
        "--structure", os.path.join(config["_base_dir"], truth["structure"]),
        "--system", config["system"],
        "--chains", *chain_args(truth["chains"]),
        "--top-n", str(options["top_n"]),
        "--top-n-intra", str(options["top_n_intra"]),
        "--residue-types", options["residue_types"],
        "--min-seq-sep", str(options["min_seq_sep"]),
        "--force-constant", str(options["force_constant"]),
        "--output", output,
    ]
    if options["wildcard_copies"]:
        argv.append("--wildcard-copies")
    generate_distance_restraints.main(argv)
    return output


def make_args(config: dict, copy_number: int, distance_csv: str, case_dir: str,
              seed: int = None):
    """Build the Namespace the runners expect, for one (copy number, seed) case."""
    values = {
        **SAMPLER_DEFAULTS, **PROPOSAL_DEFAULTS,
        **config["sampler_params"], **config["proposal"],
        "system": config["system"],
        "copy_number": copy_number,
        "distance_csv": distance_csv,
        "prior": config["prior"],
        "prior_box_half_width": config.get("prior_box_half_width", 300.0),
        "seed": config["seed"] if seed is None else seed,
        "output_dir": case_dir,
        "run_name": f"n{copy_number}" if seed is None else f"n{copy_number}_s{seed}",
        "debug": False, "debug_every": 1, "quiet": True,
    }
    return argparse.Namespace(**values)


def rmf_path_for(sampler: str, out_prefix: str) -> str:
    """Where each runner leaves its trajectory."""
    if sampler == "imp_rex":
        return os.path.join(f"{out_prefix}_imp_rex", "rmfs", "0.rmf3")
    return f"{out_prefix}_{sampler}.rmf3"


def reference_coordinates(config: dict, copy_number: int) -> Dict[int, np.ndarray]:
    """Ground-truth bead positions for each copy, from *this case's* structure.

    Every case must be scored against its own ground truth. The tetramer's
    dimers differ from the reference dimer in kcoil_ecoil.pdb by ~11 A RMSD,
    so measuring a two-copy model against the one-copy reference would report
    an 11 A floor that has nothing to do with how the sampler did.

    Returns structured-bead coordinates per copy, ordered exactly as
    evaluate_recovery.bead_coordinates orders them, so the two can be compared
    row for row. Bead centers are the plain mean over the constituent
    residues' atoms, which reproduces IMP's own placement exactly.
    """
    truth = ground_truth_for(config, copy_number)
    chains = {chain: (protein, index)
              for chain, (protein, index) in truth["chains"].items()}
    atoms = generate_distance_restraints.chain_atoms(
        os.path.join(config["_base_dir"], truth["structure"]), chains)

    # Bead decomposition is a property of the representation, so one build
    # supplies it for every copy; only the coordinates differ between copies.
    system = config["_system"]
    built, _, _ = system.build_system(copy_number=1, distance_csv=False)
    structured: Dict[str, List[tuple]] = {}
    for protein in system.PROTEINS:
        beads = []
        for particle in IMP.atom.Selection(
                built.root_hier, molecule=protein, copy_index=0,
                resolution=1).get_selected_particles():
            if not IMP.core.RigidMember.get_is_setup(particle):
                continue  # linker: no reference conformation to score against
            beads.append(tuple(int(r) for r in
                               IMP.atom.Fragment(particle).get_residue_indexes()))
        structured[protein] = beads

    reference = {}
    for copy_index in range(copy_number):
        rows = []
        for protein in system.PROTEINS:
            key = (protein, copy_index)
            if key not in atoms:
                raise ValueError(
                    f"ground truth for copy_number={copy_number} has no chain mapped to "
                    f"{protein} copy {copy_index}; check the 'chains' mapping")
            for residues in structured[protein]:
                rows.append(generate_distance_restraints.bead_center(atoms[key], residues))
        reference[copy_index] = np.asarray(rows)
    return reference


def system_size(system, copy_number: int, distance_csv: str) -> dict:
    """Sampled degrees of freedom and restraint count for one case."""
    built, _, _ = system.build_system(
        copy_number=copy_number, distance_csv=distance_csv)
    layout = dof_layout.build(built)
    constraints = distance_restraints.expand_copies(
        distance_restraints.read_distance_constraints(distance_csv), copy_number)
    return {
        "n_rigid_bodies": layout.n_rigid_bodies,
        "n_beads": len(layout.bead_particle_indexes),
        # 7 per rigid body (quaternion + translation), 3 per flexible bead.
        "n_dof": 7 * layout.n_rigid_bodies + 3 * len(layout.bead_particle_indexes),
        "n_restraints": len(constraints),
    }


def build_metric_evaluators(config: dict, reference: Dict[int, np.ndarray]):
    """One extra-metrics evaluator per copy, or None if none is configured.

    A config's `metrics_module` names a module supplying system-specific
    accuracy measurements beyond the global RMSD -- the CAPRI docking scores,
    for this example. It must expose
    `evaluators_for(system, {copy: reference_coords}) -> {copy: evaluator}`,
    where an evaluator has `.score(coords)` returning something with
    `.as_row()`. Whatever keys that row carries become columns in the results.
    """
    metrics = config.get("_metrics")
    if metrics is None:
        return None
    return metrics.evaluators_for(config["_system"], reference)


def scoring_window(n_frames: int, window: int, fraction: float) -> int:
    """First frame of the window a trajectory should be judged on.

    `window` alone is not enough. It assumes a trajectory long relative to it,
    which is true of an MCMC chain (20000 frames) and false of an SMC run,
    which writes only its best particle per temperature step -- 50 frames for
    a 50-step ladder. `max(0, n - window)` then starts at frame 0 and averages
    the whole anneal, lambda = 0 included, so a run that converged perfectly
    is scored partly on its own random starting population.

    So the window is also capped at the last `fraction` of the run, which is
    the "never average over the burn-in" rule stated in a way that does not
    depend on how many frames a sampler happens to emit. At least one frame is
    always kept.
    """
    by_count = n_frames - window
    by_fraction = n_frames * (1.0 - fraction)
    return max(0, min(n_frames - 1, int(max(by_count, by_fraction))))


def analyse_trajectory(system, rmf_path: str, copy_number: int, distance_csv: str,
                       window: int, tolerance: float,
                       reference: Dict[int, np.ndarray],
                       evaluators=None, window_fraction: float = 0.25) -> Optional[dict]:
    """Per-frame RMSD, CPU-IMP score and restraint satisfaction, over a window."""
    if not os.path.exists(rmf_path):
        return None

    built, score_function, output_objects = system.build_system(
        copy_number=copy_number, distance_csv=distance_csv)
    restraints = [r for r in output_objects
                  if isinstance(r, IMP.pmi.restraints.basic.DistanceRestraint)]
    constraints = distance_restraints.expand_copies(
        distance_restraints.read_distance_constraints(distance_csv), copy_number)

    handle = RMF.open_rmf_file_read_only(rmf_path)
    IMP.rmf.create_hierarchies(handle, built.model)
    IMP.rmf.link_hierarchies(handle, [built.root_hier])

    n_frames = handle.get_number_of_frames()
    if not n_frames:
        return None
    start = scoring_window(n_frames, window, window_fraction)

    rmsds, scores, satisfied = [], [], []
    extra: Dict[str, list] = {}
    for frame in range(start, n_frames):
        IMP.rmf.load_frame(handle, RMF.FrameID(frame))
        # Worst copy in the frame: each is scored against its own copy of the
        # ground truth, since cross-copy restraints fix which copy is which.
        rmsds.append(max(
            evaluate_recovery.superposed_rmsd(
                evaluate_recovery.bead_coordinates(built.root_hier, index, system=system),
                reference[index])
            for index in range(copy_number)))
        scores.append(float(score_function.evaluate(False)))
        # A restraint is satisfied when the two beads are within `tolerance`
        # of the distance it asks for; the score alone cannot show whether a
        # few restraints are badly violated or many are slightly off.
        #
        # The deviation is recovered from the score rather than by walking
        # back to the particles: each restraint is a harmonic well of the form
        # 0.5 * kappa * (d - d0)^2 (PMI's one-sided bounds coincide because
        # distancemin == distancemax), so |d - d0| = sqrt(2 * score / kappa).
        deviations = [
            np.sqrt(2.0 * max(r.get_restraint().get_score(), 0.0) / c.force_constant)
            for r, c in zip(restraints, constraints)]
        satisfied.append(float(np.mean([d <= tolerance for d in deviations])))

        if evaluators:
            # Score every copy, keep the worst: with copy-for-copy restraints
            # the copies are independent assemblies, and the question is
            # whether *every* one came back, not whether the best one did.
            per_copy = [
                evaluators[index].score(
                    evaluate_recovery.bead_coordinates(
                        built.root_hier, index, system=system))
                for index in range(copy_number)]
            worst = max(per_copy, key=lambda s: s.ligand_rmsd)
            for key, value in worst.as_row().items():
                extra.setdefault(key, []).append(value)

    analysis = {
        "n_frames": n_frames, "window_start": start,
        "rmsd": np.asarray(rmsds), "imp_score": np.asarray(scores),
        "satisfied": np.asarray(satisfied),
    }
    for key, values in extra.items():
        # "capri" is a string label, so it stays a plain list.
        analysis[key] = values if key == "capri" else np.asarray(values)
    return analysis


def run_sweep(config: dict, out_root: str) -> List[dict]:
    """Run every (copy number, sampler) case and measure the result."""
    import jax

    backend = jax.default_backend()
    records = []

    for copy_number in config["copy_numbers"]:
        case_dir = os.path.join(out_root, f"n{copy_number}")
        os.makedirs(case_dir, exist_ok=True)
        distance_csv = prepare_restraints(config, copy_number, case_dir)
        reference = reference_coordinates(config, copy_number)
        evaluators = build_metric_evaluators(config, reference)
        size = system_size(config["_system"], copy_number, distance_csv)
        print(f"\n=== copy_number={copy_number}: {size['n_dof']} DOF, "
              f"{size['n_restraints']} restraints ===")

        # Every seed is a different starting structure. On a multimodal
        # target -- and a docking search with nothing tethering the bodies is
        # emphatically that -- one seed measures luck as much as sampler
        # quality, so the sweep reports the spread across several.
        for seed in config["seeds"]:
            args = make_args(config, copy_number, distance_csv, case_dir, seed)
            out_prefix = os.path.join(case_dir, args.run_name)
            log_path = f"{out_prefix}.log"
            # imp_rex's runner logs through this; the BlackJAX runners take the path.
            case_logger = logging_config.configure_logging(log_path=log_path)
            if len(config["seeds"]) > 1:
                print(f"  --- seed {seed} ---")

            for sampler in config["samplers"]:
                print(f"  running {sampler} ...", flush=True)
                case_args = copy_module.copy(args)
                started = time.perf_counter()
                try:
                    elapsed, final_score, note = runner.RUNNERS[sampler](
                        sampler, case_args, out_prefix, log_path, case_logger)
                    failure = None
                except Exception as error:  # a failed sampler must not lose the sweep
                    elapsed, final_score, note = None, float("nan"), str(error)
                    failure = f"{type(error).__name__}: {error}"
                    print(f"    FAILED: {failure}")

                record = {
                    "copy_number": copy_number, "sampler": sampler,
                    "seed": seed, "backend": backend,
                    "wall_time": elapsed.wall_time if elapsed else time.perf_counter() - started,
                    "cpu_time": elapsed.cpu_time if elapsed else float("nan"),
                    "sampler_score": final_score, "note": note, "failure": failure,
                    **size,
                }
                if failure is None:
                    analysis = analyse_trajectory(
                        config["_system"],
                        rmf_path_for(sampler, out_prefix), copy_number, distance_csv,
                        config["score_window"], config["satisfaction_tolerance"],
                        reference, evaluators, config["score_window_fraction"])
                    if analysis:
                        record.update(analysis)
                        print(f"    {record['wall_time']:.1f}s | "
                              f"RMSD {analysis['rmsd'].min():.2f}-{analysis['rmsd'].max():.2f} A | "
                              f"IMP score {analysis['imp_score'].min():.1f} | "
                              f"{100*analysis['satisfied'].max():.0f}% restraints satisfied")
                records.append(record)
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="JSON sweep configuration")
    parser.add_argument("--output-dir", default=None,
                        help="override the config's output_dir")
    parser.add_argument("--skip-run", action="store_true",
                        help="re-render the report from an existing results.json")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_root = args.output_dir or os.path.join(config["_base_dir"], config["output_dir"])
    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(out_root, "results.json")

    if args.skip_run:
        import benchmark_report
        records = benchmark_report.load_records(results_path)
    else:
        records = run_sweep(config, out_root)
        import benchmark_report
        benchmark_report.save_records(results_path, records)

    import benchmark_report
    pdf_path = os.path.join(out_root, f"{config['name']}_report.pdf")
    benchmark_report.write_report(pdf_path, config, records)
    print(f"\nresults: {results_path}\nreport:  {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
