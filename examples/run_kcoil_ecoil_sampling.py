#!/usr/bin/env python
"""Run and compare samplers on the KCOIL/ECOIL toy system.

Three samplers are available, chosen with --samplers (any combination):
    rmh      BlackJAX RMH via wrapper_impjax.run_sampling (SO(3)-aware
             proposal, doc/design.tex Section 5-6).
    smc      BlackJAX fixed-schedule SMC via wrapper_impjax.run_smc_sampling
             (untuned, doc/design.tex's SMC section); only the best-scoring
             particle per temperature step is written to RMF3.
    imp_rex  IMP's own native replica-exchange Monte Carlo
             (IMP.pmi.macros.ReplicaExchange), on the exact same system and
             scoring function, as the ground-truth-implementation baseline.

Each selected sampler gets its own freshly-built system (sampling mutates
the live IMP model in place, so reusing one system across samplers would
bias whichever ran second) and is timed with impjax_toymodels.timing
(perf_counter for wall time, process_time for CPU time -- the standard
already used elsewhere in this package, see timing.py). A comparison table
is printed and written to the shared run log.

Usage
-----
    python run_kcoil_ecoil_sampling.py --copy-number 1 --samplers rmh smc imp_rex \
        --n-steps 2000 --output-dir out/

Outputs, under --output-dir, named by --run-name (default "kcoil_ecoil"):
    <run-name>_rmh.rmf3 / _stats.csv        (if "rmh" selected)
    <run-name>_smc.rmf3 / _stats.csv        (if "smc" selected; best-particle-only)
    <run-name>_imp_rex/                     (if "imp_rex" selected; IMP's own output dir)
    <run-name>.log                          all samplers share one run log
    <run-name>_score_comparison.csv         if --debug is set (rmh/smc only)
"""

import argparse
import os

import IMP.pmi.macros
import jax

from kcoil_ecoil_system import build_kcoil_ecoil_system

from impjax_toymodels import elapsed_timing, logging_config, start_timing, wrapper_impjax

SAMPLER_CHOICES = ("rmh", "smc", "imp_rex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copy-number", type=int, default=1, help="copies of each protein (KCOIL, ECOIL)")
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=SAMPLER_CHOICES,
        default=["rmh"],
        help="which sampler(s) to run and compare (default: rmh only)",
    )

    # BlackJAX RMH / SMC shared proposal knobs.
    parser.add_argument("--mode", choices=("rotation", "translation", "rigid", "beads", "all"), default="all")
    parser.add_argument("--sigma-rotation", type=float, default=0.05)
    parser.add_argument("--sigma-translation", type=float, default=1.0)
    parser.add_argument("--sigma-bead", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0, help="JAX PRNG seed")

    # RMH-specific.
    parser.add_argument("--n-steps", type=int, default=2000, help="RMH: number of MCMC steps")
    parser.add_argument("--burnin", type=int, default=0)
    parser.add_argument("--thin", type=int, default=1)

    # SMC-specific (defaults match smc_fixed_schedule.py's untuned constants).
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--n-temperature-steps", type=int, default=20)
    parser.add_argument("--n-mcmc-steps", type=int, default=10, help="SMC: mutation sweeps per temperature step")
    parser.add_argument("--schedule", choices=("linear", "geometric", "sigmoid"), default="linear")

    # IMP replica exchange-specific.
    parser.add_argument("--imp-rex-frames", type=int, default=2000)
    parser.add_argument("--imp-rex-mc-steps", type=int, default=5, help="MC steps per replica-exchange frame")
    parser.add_argument("--imp-rex-max-temp", type=float, default=4.0)

    parser.add_argument("--debug", action="store_true", help="verify JAX vs CPU-IMP scores (rmh/smc only)")
    parser.add_argument("--output-dir", default="out")
    parser.add_argument("--run-name", default="kcoil_ecoil")
    parser.add_argument("--quiet", action="store_true", help="suppress per-step console progress")
    return parser.parse_args()


def _run_blackjax_rmh(args, out_prefix: str, log_path: str):
    built, score_function, _ = build_kcoil_ecoil_system(copy_number=args.copy_number)
    score_function.evaluate(False)  # materialize IMP's JAX export

    timer = start_timing()
    positions, log_probs, acceptance_rate = wrapper_impjax.run_sampling(
        built,
        score_function,
        jax.random.PRNGKey(args.seed),
        n_steps=args.n_steps,
        mode=args.mode,
        sigma_rotation=args.sigma_rotation,
        sigma_translation=args.sigma_translation,
        sigma_bead=args.sigma_bead,
        burnin=args.burnin,
        thin=args.thin,
        rmf_path=f"{out_prefix}_rmh.rmf3",
        log_path=log_path,
        debug=args.debug,
        verbose=not args.quiet,
    )
    elapsed = elapsed_timing(timer)
    return elapsed, float(log_probs[-1]) if len(log_probs) else float("nan"), f"acceptance={100*acceptance_rate:.1f}%"


def _run_blackjax_smc(args, out_prefix: str, log_path: str):
    built, score_function, _ = build_kcoil_ecoil_system(copy_number=args.copy_number)
    score_function.evaluate(False)

    timer = start_timing()
    best_thetas, best_scores, _ = wrapper_impjax.run_smc_sampling(
        built,
        score_function,
        jax.random.PRNGKey(args.seed),
        n_particles=args.n_particles,
        n_temperature_steps=args.n_temperature_steps,
        n_mcmc_steps=args.n_mcmc_steps,
        schedule=args.schedule,
        mode=args.mode,
        sigma_rotation=args.sigma_rotation,
        sigma_translation=args.sigma_translation,
        sigma_bead=args.sigma_bead,
        rmf_path=f"{out_prefix}_smc.rmf3",
        log_path=log_path,
        debug=args.debug,
        verbose=not args.quiet,
    )
    elapsed = elapsed_timing(timer)
    return elapsed, best_scores[-1], f"particles={args.n_particles}"


def _run_imp_replica_exchange(args, out_prefix: str, run_logger):
    """IMP's own native sampler (ground-truth-implementation baseline), on a
    freshly built copy of the exact same system and scoring function."""
    built, score_function, output_objects = build_kcoil_ecoil_system(copy_number=args.copy_number)
    output_dir = f"{out_prefix}_imp_rex"

    timer = start_timing()
    rex = IMP.pmi.macros.ReplicaExchange(
        built.model,
        root_hier=built.root_hier,
        monte_carlo_sample_objects=built.dof.get_movers(),
        replica_exchange_maximum_temperature=args.imp_rex_max_temp,
        global_output_directory=output_dir,
        output_objects=output_objects,
        nframes_write_coordinates=1,
        monte_carlo_steps=args.imp_rex_mc_steps,
        number_of_frames=args.imp_rex_frames,
        number_of_best_scoring_models=1,
    )
    rex.execute_macro()
    elapsed = elapsed_timing(timer)

    final_score = score_function.evaluate(False)
    run_logger.info(
        "imp_rex finished: %d frames x %d MC steps in %.2fs wall / %.2fs cpu, "
        "final-frame IMP score=%.2f, output=%s",
        args.imp_rex_frames,
        args.imp_rex_mc_steps,
        elapsed.wall_time,
        elapsed.cpu_time,
        final_score,
        output_dir,
    )
    return elapsed, float(final_score), f"frames={args.imp_rex_frames}"


RUNNERS = {
    "rmh": lambda args, out_prefix, log_path, run_logger: _run_blackjax_rmh(args, out_prefix, log_path),
    "smc": lambda args, out_prefix, log_path, run_logger: _run_blackjax_smc(args, out_prefix, log_path),
    "imp_rex": lambda args, out_prefix, log_path, run_logger: _run_imp_replica_exchange(args, out_prefix, run_logger),
}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_prefix = os.path.join(args.output_dir, args.run_name)
    log_path = f"{out_prefix}.log"
    run_logger = logging_config.configure_logging(log_path=log_path)

    run_logger.info("Comparing samplers %s on KCOIL/ECOIL (copy_number=%d)", args.samplers, args.copy_number)

    results = {}
    for sampler in args.samplers:
        print(f"--- running {sampler} ---")
        elapsed, score, note = RUNNERS[sampler](args, out_prefix, log_path, run_logger)
        results[sampler] = (elapsed, score, note)
        run_logger.info(
            "sampler=%s wall=%.2fs cpu=%.2fs score=%.2f (%s)",
            sampler,
            elapsed.wall_time,
            elapsed.cpu_time,
            score,
            note,
        )

    header = f"{'sampler':<10} {'wall (s)':>10} {'cpu (s)':>10} {'score':>16}  note"
    print(header)
    run_logger.info(header)
    for sampler, (elapsed, score, note) in results.items():
        row = f"{sampler:<10} {elapsed.wall_time:>10.2f} {elapsed.cpu_time:>10.2f} {score:>16.2f}  {note}"
        print(row)
        run_logger.info(row)

    print(f"\nlog: {log_path}")


if __name__ == "__main__":
    main()
