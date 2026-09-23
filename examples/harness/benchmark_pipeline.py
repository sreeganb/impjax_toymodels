"""Sweep copy numbers, seeds and samplers; report accuracy, convergence and time.

A standalone harness over everything else in examples/: it builds no systems
and runs no samplers of its own, it drives run_sampling_comparison.py's
runners and generate_distance_restraints.py's selection, then measures what
came out and writes a single multi-page PDF.  Which system is swept comes
from the config's "system" key, so this file names no molecules.

    python harness/benchmark_pipeline.py --config data/benchmark_config.json

What it measures (trajectory_analysis.py does all of it, identically for every
sampler, from the RMF3 files each run wrote):

* **RMSD of the best-scoring models.** Every frame is re-scored by IMP on the
  CPU with the full scoring function, the `n_best_models` lowest-scoring
  frames after `burnin_fraction` are kept, and their RMSD to the ground truth
  -- the system as built, before the shuffle -- is reported. The RMSD is
  IMP's own `IMP.pmi.analysis.Precision`, the same primitive PMI_analysis's
  accuracy.py uses (structure_rmsd.py). Samplers are judged on the models you
  would actually pick.
* **Score convergence**: the best IMP score found so far against wall-clock
  time.
* **Time to solution**: wall seconds until the run's best-scoring model is
  within `success_rmsd` of the ground truth. This is the headline number: it
  folds sampling efficiency and hardware (CPU vs GPU, 1 vs N replicas) into
  the one quantity a modeller pays for.

Replica exchange runs `imp_rex_replicas` replicas under `imp_rex_launcher`
(default "auto": the mpiexec in this Python's env, i.e. the MPI IMP.mpi was
built with) -- see run_imp_rex.py.

Seeds
-----
A config's "seeds" list runs the whole sweep once per seed, and each seed is
a different starting structure shared by every sampler (run_sampling_
comparison.seed_imp seeds the RNG `shuffle_configuration` draws from). On a
multimodal target one seed measures luck as much as sampler quality.
"""


import argparse
import copy as copy_module
import glob
import json
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_distance_restraints
import run_sampling_comparison as runner
import structure_rmsd
import system_registry
import trajectory_analysis
from impjax_toymodels import contact_map, dof_layout, distance_restraints, logging_config

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Filled in from the JSON, but every key needs a value before a Namespace can
#: be handed to run_sampling_comparison's runners.
SAMPLER_DEFAULTS = {
    # burnin is 0 so every step is written: the convergence plot needs the
    # whole run, and burn-in is applied once, in the analysis (burnin_fraction).
    "n_steps": 5000, "burnin": 0, "thin": 1,
    "n_particles": 100, "n_temperature_steps": 100, "n_mcmc_steps": 10,
    "schedule": "linear", "target_ess": 0.5,
    "imp_rex_frames": 5000, "imp_rex_mc_steps": 10, "imp_rex_max_temp": 2.5,
    "imp_rex_replicas": 1, "imp_rex_launcher": "auto",
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
    config.setdefault("name", "benchmark")
    config.setdefault("output_dir", "out/benchmark")
    # The accuracy measurement: the n_best_models lowest-scoring frames of a
    # run, after dropping the first burnin_fraction of each trajectory.
    config.setdefault("n_best_models", 50)
    config.setdefault("burnin_fraction", 0.25)
    # A run is "solved" once its best-scoring model is this close (A).
    config.setdefault("success_rmsd", 5.0)
    config.setdefault("seed", 0)
    # A list of seeds runs the whole sweep once per seed. Each seed is a
    # different starting structure (see run_sampling_comparison.seed_imp), so
    # this is what turns a single anecdote into a distribution.
    config.setdefault("seeds", [config["seed"]])
    config.setdefault("prior", "connectivity+box")
    config.setdefault("restraints", {})
    config.setdefault("proposal", {})
    config.setdefault("sampler_params", {})
    return config


def ground_truth_for(config: dict, copy_number: int) -> dict:
    """Resolve which structure and contact map the *restraints* come from.

    Only restraint generation reads this. The RMSD reference is always the
    unshuffled build of the system itself (structure_rmsd.write_reference).

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


def trajectory_files(sampler: str, out_prefix: str, wall_time: float):
    """Every RMF3 file a run left, and the stretch of the run each one spans.

    rmh      one chain, frames evenly spread over the run.
    smc*     the best particle per temperature step (the anneal, spread over
             the run) plus the final population (independent samples, all
             produced at the end, no burn-in).
    imp_rex  one file per replica; replicas run concurrently, so each spans
             the whole run.
    """
    if sampler == "imp_rex":
        paths = sorted(glob.glob(os.path.join(f"{out_prefix}_imp_rex", "rmfs", "*.rmf3")))
        return [trajectory_analysis.TrajectoryFile(p, 0.0, wall_time) for p in paths]
    files = [trajectory_analysis.TrajectoryFile(f"{out_prefix}_{sampler}.rmf3", 0.0, wall_time)]
    if sampler.startswith("smc"):
        files.append(trajectory_analysis.TrajectoryFile(
            f"{out_prefix}_{sampler}_population.rmf3", wall_time, wall_time, has_burnin=False))
    return files


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


def _median(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def run_sweep(config: dict, out_root: str) -> List[dict]:
    """Run every (copy number, sampler) case and measure the result."""
    import jax

    backend = jax.default_backend()
    records = []

    for copy_number in config["copy_numbers"]:
        case_dir = os.path.join(out_root, f"n{copy_number}")
        os.makedirs(case_dir, exist_ok=True)
        distance_csv = prepare_restraints(config, copy_number, case_dir)
        # Ground truth = this system, as built, before the shuffle, written
        # out as a one-frame RMF3 so IMP's own Precision can measure against
        # it (and so it can be opened in Chimera beside any model).
        reference = structure_rmsd.write_reference(
            config["_system"], copy_number, distance_csv,
            os.path.join(case_dir, "reference.rmf3"))
        size = system_size(config["_system"], copy_number, distance_csv)
        print(f"\n=== copy_number={copy_number}: {size['n_dof']} DOF, "
              f"{size['n_restraints']} restraints, ground-truth IMP score "
              f"{reference.imp_score:.2f} ===")

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
                    elapsed, _, note = runner.RUNNERS[sampler](
                        sampler, case_args, out_prefix, log_path, case_logger)
                    failure = None
                except Exception as error:  # a failed sampler must not lose the sweep
                    elapsed, note = None, str(error)
                    failure = f"{type(error).__name__}: {error}"
                    print(f"    FAILED: {failure}")

                record = {
                    "copy_number": copy_number, "sampler": sampler,
                    "seed": seed, "backend": backend,
                    "wall_time": elapsed.wall_time if elapsed else time.perf_counter() - started,
                    "cpu_time": elapsed.cpu_time if elapsed else float("nan"),
                    "note": note, "failure": failure,
                    "reference_score": reference.imp_score, **size,
                }
                if failure is None:
                    record.update(trajectory_analysis.analyse_run(
                        config["_system"],
                        trajectory_files(sampler, out_prefix, record["wall_time"]),
                        copy_number, distance_csv, reference,
                        config["n_best_models"], config["burnin_fraction"],
                        config["success_rmsd"]))
                    solved = record["time_to_solution"]
                    print(f"    {record['wall_time']:.1f}s, {record['n_frames']} frames | "
                          f"best-{len(record['best_rmsds'])} RMSD median "
                          f"{_median(record['best_rmsds']):.2f} A | "
                          + (f"solved at {solved:.1f}s" if np.isfinite(solved) else "not solved"))
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
