#!/usr/bin/env python
"""IMP's native replica-exchange Monte Carlo, one replica per MPI rank.

This is the baseline the BlackJAX samplers are measured against, so it is run
the way IMP users actually run it: many replicas in parallel under MPI,

    mpirun -np 8 python harness/run_imp_rex.py --system docking_system \\
        --output-dir out/n1_s0_imp_rex --seed 0 --frames 60000 &> output_rex.out

Each rank is one replica. IMP.pmi.macros.ReplicaExchange builds a geometric
temperature ladder from 1 to --max-temp across however many ranks there are,
and every rank writes its own trajectory, rmfs/<replica>.rmf3, and its own
stat.<replica>.out.

The launcher must belong to the same MPI that IMP.mpi was built against. By
default (`resolve_launcher("auto")`) the benchmark uses the mpiexec installed
next to this Python, which in a conda env is exactly that one -- so do NOT
rely on a `module load`ed Open MPI mpirun to launch a conda (MPICH) IMP.

benchmark_pipeline.py launches exactly this command when a config sets
"imp_rex_replicas" > 1 (see `launch`), so a sweep on the lab machine needs no
manual step. With one replica it runs in-process instead, with no MPI at all.

Two properties matter for a fair comparison:

* Every replica starts from the **same** structure as every BlackJAX sampler:
  IMP's RNG is seeded with --seed before the build shuffles the system. Only
  after the build is each rank's generator re-seeded to its own stream, so the
  replicas' Monte Carlo moves are independent.
* Timing covers sampling only (not Python start-up or the build) and is
  written per rank to timing.<replica>.json. The run's wall time is the
  slowest rank's; its CPU time is the sum over ranks.

Always scored with the system's *combined* scoring function: IMP has no notion
of a tempered likelihood versus an untempered prior.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

import IMP
import IMP.pmi.macros
import IMP.pmi.samplers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import system_registry

HERE = os.path.abspath(__file__)

#: Environment variables the common MPI launchers set to the job size. If any
#: is > 1 but IMP was built without IMP.mpi, every rank would run the same
#: single replica and all of them would write rmfs/0.rmf3 over one another.
MPI_SIZE_VARIABLES = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS")


def launched_ranks() -> int:
    """How many ranks the MPI launcher started, or 1 if not under MPI."""
    for name in MPI_SIZE_VARIABLES:
        if os.environ.get(name, "").isdigit():
            return int(os.environ[name])
    return 1


def replica_exchange_object():
    """IMP.mpi's replica-exchange object, or None to let PMI run one replica."""
    try:
        import IMP.mpi
    except ImportError:
        if launched_ranks() > 1:
            raise RuntimeError(
                f"launched on {launched_ranks()} MPI ranks but this IMP has no IMP.mpi "
                "module -- load an MPI-enabled IMP build (e.g. `module load imp`) first")
        return None
    return IMP.mpi.ReplicaExchange()


def resolve_launcher(setting: str = "auto") -> list:
    """The MPI launcher command, as an argv prefix.

    "auto" (the default) picks the `mpiexec` installed next to this Python --
    in a conda env, that is the launcher of the MPI that IMP.mpi was linked
    against. That matters: an MPICH-built IMP (conda-forge's default) cannot
    start under Open MPI's `mpirun`, and fails in MPI_Init with "launcher not
    compatible with PMI1 client". A `module load`ed Open MPI puts exactly that
    mpirun first on PATH. Falls back to `mpirun` on PATH. Any other value is
    used as given, e.g. "mpirun --oversubscribe".
    """
    if setting and setting != "auto":
        return shlex.split(setting)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(os.path.dirname(sys.executable), name)
        if os.access(candidate, os.X_OK):
            return [candidate]
    return ["mpirun"]


#: Launcher output that means the launcher and IMP.mpi's MPI do not match.
MISMATCH_SIGNS = ("launcher not compatible with PMI", "PMI_Init returned", "pmi1_init")


def run(system_name: str, copy_number: int, distance_csv, seed: int, frames: int,
        mc_steps: int, max_temp: float, output_dir: str) -> dict:
    """Run this rank's replica and write its timing file. Returns the timing."""
    system = system_registry.resolve(system_name)
    rem = replica_exchange_object()
    replica = rem.get_my_index() if rem is not None else 0

    # Same seed on every rank -> the same shuffled start for every replica.
    IMP.random_number_generator.seed(seed)
    built, _, output_objects = system.build_system(
        copy_number=copy_number, distance_csv=distance_csv)
    # Now give each replica its own Monte Carlo stream.
    IMP.random_number_generator.seed(seed * 1000 + replica + 1)

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    IMP.pmi.macros.ReplicaExchange(
        built.model,
        root_hier=built.root_hier,
        monte_carlo_sample_objects=built.dof.get_movers(),
        replica_exchange_maximum_temperature=max_temp,
        replica_exchange_object=rem,
        global_output_directory=output_dir,
        output_objects=output_objects,
        nframes_write_coordinates=1,
        monte_carlo_steps=mc_steps,
        number_of_frames=frames,
        number_of_best_scoring_models=0,
    ).execute_macro()
    timing = {"replica": replica, "wall_time": time.perf_counter() - wall_start,
              "cpu_time": time.process_time() - cpu_start}
    with open(os.path.join(output_dir, f"timing.{replica}.json"), "w") as handle:
        json.dump(timing, handle)
    return timing


def command(args, output_dir: str) -> list:
    """The command line that runs one replica of this case."""
    argv = [sys.executable, HERE, "--system", args.system,
            "--copy-number", str(args.copy_number), "--seed", str(args.seed),
            "--frames", str(args.imp_rex_frames), "--mc-steps", str(args.imp_rex_mc_steps),
            "--max-temp", str(args.imp_rex_max_temp), "--output-dir", output_dir]
    if getattr(args, "distance_csv", None):
        argv += ["--distance-csv", args.distance_csv]
    return argv


def launch(args, output_dir: str) -> dict:
    """Run a whole replica-exchange case and return its combined timing.

    `args.imp_rex_replicas` > 1 runs `<launcher> -np N python run_imp_rex.py
    ...` as a subprocess, with its stdout/stderr going to <output_dir>.out --
    the equivalent of `mpirun -np N python run_imp_rex.py &> output_rex.out`.
    One replica runs in this process. The output directory is cleared first so
    no replica file from an earlier, larger run is mistaken for this one's.
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    replicas = int(getattr(args, "imp_rex_replicas", 1))
    if replicas <= 1:
        run(args.system, args.copy_number, getattr(args, "distance_csv", None), args.seed,
            args.imp_rex_frames, args.imp_rex_mc_steps, args.imp_rex_max_temp, output_dir)
    else:
        launcher = resolve_launcher(getattr(args, "imp_rex_launcher", "auto"))
        argv = launcher + ["-np", str(replicas)] + command(args, output_dir)
        log_path = f"{output_dir}.out"
        print(f"    {shlex.join(argv)} &> {log_path}", flush=True)
        with open(log_path, "w") as log:
            completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            with open(log_path, errors="replace") as log:
                mismatch = any(sign in log.read() for sign in MISMATCH_SIGNS)
            hint = (f" -- {launcher[0]} belongs to a different MPI than the one IMP.mpi "
                    "was built with; set imp_rex_launcher to \"auto\" (this env's own "
                    "mpiexec) or to that MPI's launcher" if mismatch else "")
            raise RuntimeError(
                f"replica exchange exited {completed.returncode}; see {log_path}{hint}")

    timings = []
    for name in sorted(os.listdir(output_dir)):
        if name.startswith("timing.") and name.endswith(".json"):
            with open(os.path.join(output_dir, name)) as handle:
                timings.append(json.load(handle))
    if len(timings) != replicas:
        raise RuntimeError(f"expected {replicas} replica timing files in {output_dir}, "
                           f"found {len(timings)}")
    return {"wall_time": max(t["wall_time"] for t in timings),
            "cpu_time": sum(t["cpu_time"] for t in timings), "replicas": replicas}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    system_registry.add_argument(parser)
    parser.add_argument("--copy-number", type=int, default=1)
    parser.add_argument("--distance-csv", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="same on every rank: fixes the shared starting structure")
    parser.add_argument("--frames", type=int, default=2000, help="replica-exchange frames")
    parser.add_argument("--mc-steps", type=int, default=10, help="MC steps per frame")
    parser.add_argument("--max-temp", type=float, default=4.0,
                        help="top of the temperature ladder (bottom is 1)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    run(args.system, args.copy_number, args.distance_csv, args.seed, args.frames,
        args.mc_steps, args.max_temp, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
