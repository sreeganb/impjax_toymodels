# examples

Example scripts and notebooks for IMP/JAX/BlackJAX toy-model experiments live
in this folder. Unlike `src/impjax_toymodels/`, code here is system-specific
and not held to the package's 200-300 line/file budget -- it exists to show
one particular system going through the (system-agnostic) wrapper, not to be
reused as a library.

## KCOIL/ECOIL sampling

`kcoil_ecoil_system.py` builds the toy coiled-coil dimer used in
`test/test_imp_system.py` (JSON + "domains" degrees-of-freedom: two
structured rigid-body domains per protein copy, connected by a flexible
linker), using the data files copied into `examples/data/`.

`run_kcoil_ecoil_sampling.py` runs it through the full pipeline -- build,
sample, write RMF3 trajectory + stat file + log -- in one command, and can
compare multiple samplers on the same system:

```bash
# one sampler
python run_kcoil_ecoil_sampling.py --copy-number 2 --n-steps 2000 --output-dir out/

# compare BlackJAX RMH, BlackJAX fixed-schedule SMC, and IMP's own native
# replica exchange -- each gets its own freshly-built system (fairness),
# each is timed with impjax_toymodels.timing, and a comparison table
# (wall/cpu time, final/best score) is printed and logged
python run_kcoil_ecoil_sampling.py --copy-number 2 --samplers rmh smc imp_rex --output-dir out/

# verify the JAX-exported score matches IMP's own CPU score periodically
# (rmh/smc only) -- writes <run-name>_score_comparison.csv
python run_kcoil_ecoil_sampling.py --samplers rmh --debug --output-dir out/
```

See `python run_kcoil_ecoil_sampling.py --help` for all options (sampling
mode, proposal scales, SMC particle/temperature-step counts, IMP replica-
exchange frame counts, burn-in/thinning, RNG seed, output naming). Outputs
land in `--output-dir`, named by `--run-name` (default `kcoil_ecoil`):
`<run-name>_rmh.rmf3`/`_stats.csv`, `<run-name>_smc.rmf3`/`_stats.csv`
(best-scoring particle per temperature step only), `<run-name>_imp_rex/`
(IMP's own output directory), and one shared `<run-name>.log` for whichever
samplers were run.
