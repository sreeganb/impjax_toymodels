#!/usr/bin/env python
"""Run the full impjax_toymodels pipeline on the KCOIL/ECOIL toy system.

Build -> sample -> RMF3 trajectory + stat file + log, in one call to
wrapper_impjax.run_sampling. This is the runnable example requested
alongside the wrapper: everything system-specific (KCOIL/ECOIL, copy
number, restraint choice) lives here and in kcoil_ecoil_system.py; the
sampling itself is entirely handled by the system-agnostic
src/impjax_toymodels package.

Usage
-----
    python run_kcoil_ecoil_sampling.py --copy-number 2 --n-steps 2000 \
        --mode all --output-dir out/

Outputs (under --output-dir, named by --run-name, default "kcoil_ecoil"):
    <run-name>.rmf3        trajectory, one frame per saved sample
    <run-name>_stats.csv   step, log_prob per saved sample
    <run-name>.log         full run log (also printed to the console)
"""

import argparse
import os
import IMP.pmi.macros

import jax

from kcoil_ecoil_system import build_kcoil_ecoil_system

from impjax_toymodels import wrapper_impjax
from impjax_toymodels import dof_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copy-number", type=int, default=1, help="copies of each protein (KCOIL, ECOIL)")
    parser.add_argument("--n-steps", type=int, default=6000, help="number of MCMC steps")
    parser.add_argument(
        "--mode",
        choices=("rotation", "translation", "rigid", "beads", "all"),
        default="all",
        help="which degrees of freedom to sample (see dof_layout.SAMPLING_MODES)",
    )
    parser.add_argument("--sigma-rotation", type=float, default=0.05, help="rigid-body rotation proposal scale")
    parser.add_argument("--sigma-translation", type=float, default=1.0, help="rigid-body translation proposal scale")
    parser.add_argument("--sigma-bead", type=float, default=1.0, help="flexible-bead proposal scale")
    parser.add_argument("--burnin", type=int, default=0)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="JAX PRNG seed")
    parser.add_argument("--output-dir", default="out", help="directory for the .rmf3/.csv/.log outputs")
    parser.add_argument("--run-name", default="kcoil_ecoil", help="base filename for the outputs")
    parser.add_argument("--quiet", action="store_true", help="suppress per-step console progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rmf_path = os.path.join(args.output_dir, f"{args.run_name}.rmf3")
    log_path = os.path.join(args.output_dir, f"{args.run_name}.log")
    output_objects = []
    
    built, score_function, output_objects = build_kcoil_ecoil_system(copy_number=args.copy_number)
    initial_score = score_function.evaluate(False)  # materializes IMP's JAX export
    print(f"Built KCOIL/ECOIL system: copy_number={args.copy_number}, initial IMP score={initial_score:.4f}")

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
        rmf_path=rmf_path,
        log_path=log_path,
        verbose=not args.quiet,
    )

    print(f"Done: {len(positions)} sample(s) saved, acceptance rate {100 * acceptance_rate:.1f}%")
    print(f"  trajectory : {rmf_path}")
    print(f"  stats      : {os.path.splitext(rmf_path)[0]}_stats.csv")
    print(f"  log        : {log_path}")
    
    """
    Now run the IMP replica exchange sampler on the exact same system and the exact same
    scoring functions. This should give you more context on what happens in 6000 steps.
    The output from built contains the information needed, but here we have to define some 
    other necessary parameters.
    """
    # calculate the time taken for the IMP replica exchange sampling
    import time
    start_time = time.time()
    print("what is built: ", built)
    layout = dof_layout.build(built)
    print("what is this layout: ", built.dof)
    # Obviously start from random configurations
    mol_names = []
    for k, ms in built.molecules.items():
        mol_names += ms
    IMP.pmi.tools.shuffle_configuration(mol_names,
                                        max_translation=200,
                                        avoidcollision_rb=True)

    #built.dof.optimize_flexible_beads(100)
    rex=IMP.pmi.macros.ReplicaExchange(built.model,
                                    root_hier=built.root_hier,           
                                    monte_carlo_sample_objects=built.dof.get_movers(),
                                    replica_exchange_maximum_temperature=4.0,
                                    global_output_directory="output_new/",
                                    output_objects=output_objects,
                                    nframes_write_coordinates=1,
                                    monte_carlo_steps=5,
                                    number_of_frames=6000,
                                    number_of_best_scoring_models=1)

    rex.execute_macro()

if __name__ == "__main__":
    main()
