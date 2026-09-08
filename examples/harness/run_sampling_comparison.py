#!/usr/bin/env python
"""Run and compare samplers on any system module.

Which system is built is chosen with --system (see
examples/harness/system_registry.py for the protocol a system module
satisfies); nothing in this file names a particular molecule.

Five samplers are available, chosen with --samplers (any combination):
    rmh           BlackJAX RMH via wrapper_impjax.run_sampling (SO(3)-aware
                  proposal, doc/design.tex Section 5-6).
    smc           BlackJAX fixed-schedule SMC over `blackjax.smc.base`
                  (untuned; rebuilds and retraces its mutation kernel every
                  temperature step, which is why it is the slowest).
    smc_tempered  The same fixed ladder over `blackjax.smc.tempered`, whose
                  kernel is built and JIT-compiled once.
    smc_adaptive  `blackjax.smc.adaptive_tempered`: the ladder is solved for
                  at each step to hold the effective sample size at
                  --target-ess, so the run takes the temperature steps it
                  needs rather than a fixed count.
    imp_rex       IMP's own native replica-exchange Monte Carlo
                  (IMP.pmi.macros.ReplicaExchange), on the exact same system
                  and scoring function, as the ground-truth baseline.

All three SMC variants write only the best-scoring particle per temperature
step to RMF3.

--prior selects the prior p0(theta). It matters most for the SMC variants,
whose lambda = 0 distribution *is* the prior:
    flat              log p0 = 0 (the historical default); particles are
                      seeded by perturbing the built model.
    box               a soft bounding box, which can be sampled from, so the
                      initial population is a genuine draw from the prior.
    split             the system module's own `build_split` partition: its
                      prior scoring function becomes p0, leaving the rest as
                      the tempered likelihood. Which restraints those are is
                      the system's decision -- connectivity for a chain-
                      connected system (keeping every particle sequence-
                      connected at every temperature instead of spending the
                      anneal repairing broken chains), excluded volume for a
                      rigid-body docking problem, where connectivity is a
                      constant and carries no information.
    split+box         both, usually what you want: the split for structure,
                      the box for a samplable initial population.

  "connectivity" and "connectivity+box" are accepted as legacy aliases for
  "split" and "split+box" -- they were this option's names from when the
  harness only ever built KCOIL/ECOIL.

Each selected sampler gets its own freshly-built system (sampling mutates
the live IMP model in place, so reusing one system across samplers would
bias whichever ran second) and is timed with impjax_toymodels.timing
(perf_counter for wall time, process_time for CPU time -- the standard
already used elsewhere in this package, see timing.py). A comparison table
is printed and written to the shared run log.

Usage
-----
    python harness/run_sampling_comparison.py --system kcoil_ecoil_system \
        --copy-number 1 --samplers smc smc_tempered smc_adaptive \
        --prior split+box --output-dir out/

Outputs, under --output-dir, named by --run-name (default "kcoil_ecoil"):
    <run-name>_<sampler>.rmf3 / _stats.csv  per BlackJAX sampler selected
    <run-name>_imp_rex/                     (if "imp_rex" selected; IMP's own output dir)
    <run-name>.log                          all samplers share one run log
    <run-name>_score_comparison.csv         if --debug is set (BlackJAX samplers only)
"""

import argparse
import os

import IMP.pmi.macros
import IMP.pmi.restraints.basic
import jax

import system_registry

from impjax_toymodels import elapsed_timing, logging_config, priors, start_timing, wrapper_impjax

SAMPLER_CHOICES = ("rmh", "smc", "smc_tempered", "smc_adaptive", "imp_rex")
PRIOR_CHOICES = ("flat", "box", "split", "split+box")

#: "connectivity" was the KCOIL-specific name for "use the system's prior
#: scoring function", from before the harness served more than one system.
#: Which restraints that scoring function holds is a per-system decision --
#: connectivity for a chain-connected system, excluded volume for a rigid-body
#: docking problem -- so the generic name is "split".  The old names are kept
#: as aliases so existing command lines and config files keep working.
PRIOR_ALIASES = {"connectivity": "split", "connectivity+box": "split+box"}


def resolve_prior_name(name: str) -> str:
    """Normalize a --prior value, translating the legacy KCOIL-specific names."""
    return PRIOR_ALIASES.get(name, name)

# Maps a --samplers name to the variant name wrapper_impjax.run_smc_sampling uses.
SMC_VARIANTS = {"smc": "fixed", "smc_tempered": "tempered", "smc_adaptive": "adaptive"}


def build_system_and_prior(args):
    """Build a fresh system plus the prior selected by --prior.

    Returns (built, score_function, prior, output_objects), where
    `score_function` holds whatever restraints are being treated as the
    *likelihood*: everything, unless a split prior was asked for, in which
    case the system module's `build_split` decides which restraints become
    the untempered prior and only the rest is tempered.
    """
    system = system_registry.resolve(getattr(args, "system", None))
    box = priors.bounding_box(half_width=args.prior_box_half_width)
    # None means "the default file"; benchmark_pipeline.py points this at a
    # per-copy-number restraint set it generated for the case being run.
    distance_csv = getattr(args, "distance_csv", None)
    prior_name = resolve_prior_name(args.prior)

    if prior_name in ("split", "split+box"):
        built, likelihood_sf, prior_sf, output_objects = system.build_split(
            copy_number=args.copy_number, distance_csv=distance_csv
        )
        restraint = priors.restraint_prior(prior_sf)
        prior = restraint if prior_name == "split" else priors.composite(restraint, box)
        return built, likelihood_sf, prior, output_objects

    built, score_function, output_objects = system.build_system(
        copy_number=args.copy_number, distance_csv=distance_csv)
    return built, score_function, (None if prior_name == "flat" else box), output_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    system_registry.add_argument(parser)
    parser.add_argument("--copy-number", type=int, default=1, help="copies of each protein")
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=SAMPLER_CHOICES,
        default=["rmh"],
        help="which sampler(s) to run and compare (default: rmh only)",
    )

    parser.add_argument(
        "--prior",
        choices=PRIOR_CHOICES + tuple(PRIOR_ALIASES),
        default="flat",
        help="prior p0(theta); see module docstring (default: flat)",
    )
    parser.add_argument(
        "--prior-box-half-width",
        type=float,
        default=200.0,
        help="half-side of the bounding-box prior, for --prior box/connectivity+box",
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
    parser.add_argument(
        "--target-ess",
        type=float,
        default=0.5,
        help="smc_adaptive only: effective sample size to hold, as a fraction of the population",
    )

    # IMP replica exchange-specific.
    parser.add_argument("--imp-rex-frames", type=int, default=2000)
    parser.add_argument("--imp-rex-mc-steps", type=int, default=5, help="MC steps per replica-exchange frame")
    parser.add_argument("--imp-rex-max-temp", type=float, default=4.0)

    parser.add_argument("--debug", action="store_true", help="verify JAX vs CPU-IMP scores (rmh/smc only)")
    parser.add_argument("--distance-csv", default=None,
                        help="restraint file to use instead of the default "
                             "data/distance_constraints.csv")
    parser.add_argument("--output-dir", default="out")
    parser.add_argument("--run-name", default="kcoil_ecoil")
    parser.add_argument("--quiet", action="store_true", help="suppress per-step console progress")
    return parser.parse_args()


def _run_blackjax_rmh(args, out_prefix: str, log_path: str):
    built, score_function, prior, _ = build_system_and_prior(args)
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
        prior=prior,
        burnin=args.burnin,
        thin=args.thin,
        rmf_path=f"{out_prefix}_rmh.rmf3",
        log_path=log_path,
        debug=args.debug,
        verbose=not args.quiet,
    )
    elapsed = elapsed_timing(timer)
    return elapsed, float(log_probs[-1]) if len(log_probs) else float("nan"), f"acceptance={100*acceptance_rate:.1f}%"


def _run_blackjax_smc(sampler: str, args, out_prefix: str, log_path: str):
    """Run one of the three SMC variants; `sampler` is the --samplers name."""
    built, score_function, prior, _ = build_system_and_prior(args)
    score_function.evaluate(False)

    timer = start_timing()
    best_thetas, best_scores, lambdas = wrapper_impjax.run_smc_sampling(
        built,
        score_function,
        jax.random.PRNGKey(args.seed),
        variant=SMC_VARIANTS[sampler],
        n_particles=args.n_particles,
        n_temperature_steps=args.n_temperature_steps,
        n_mcmc_steps=args.n_mcmc_steps,
        schedule=args.schedule,
        target_ess=args.target_ess,
        mode=args.mode,
        sigma_rotation=args.sigma_rotation,
        sigma_translation=args.sigma_translation,
        sigma_bead=args.sigma_bead,
        prior=prior,
        rmf_path=f"{out_prefix}_{sampler}.rmf3",
        log_path=log_path,
        debug=args.debug,
        verbose=not args.quiet,
    )
    elapsed = elapsed_timing(timer)
    return elapsed, best_scores[-1], f"particles={args.n_particles} steps={len(lambdas) - 1}"


def _run_imp_replica_exchange(args, out_prefix: str, run_logger):
    """IMP's own native sampler (ground-truth-implementation baseline), on a
    freshly built copy of the exact same system and scoring function.

    Always uses the *combined* scoring function: IMP has no notion of a
    tempered likelihood versus an untempered prior, so a --prior split would
    not be a like-for-like baseline.
    """
    system = system_registry.resolve(getattr(args, "system", None))
    built, score_function, output_objects = system.build_system(
        copy_number=args.copy_number, distance_csv=getattr(args, "distance_csv", None))
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
    "rmh": lambda name, args, out_prefix, log_path, run_logger: _run_blackjax_rmh(args, out_prefix, log_path),
    "imp_rex": lambda name, args, out_prefix, log_path, run_logger: _run_imp_replica_exchange(
        args, out_prefix, run_logger
    ),
}
# All three SMC variants share one runner, distinguished by the sampler name.
for _smc_name in SMC_VARIANTS:
    RUNNERS[_smc_name] = (
        lambda name, args, out_prefix, log_path, run_logger: _run_blackjax_smc(
            name, args, out_prefix, log_path
        )
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_prefix = os.path.join(args.output_dir, args.run_name)
    log_path = f"{out_prefix}.log"
    run_logger = logging_config.configure_logging(log_path=log_path)

    run_logger.info(
        "Comparing samplers %s on %s (copy_number=%d, prior=%s)",
        args.samplers,
        args.system,
        args.copy_number,
        resolve_prior_name(args.prior),
    )

    results = {}
    for sampler in args.samplers:
        print(f"--- running {sampler} ---")
        elapsed, score, note = RUNNERS[sampler](sampler, args, out_prefix, log_path, run_logger)
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
