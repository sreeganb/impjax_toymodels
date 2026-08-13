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
sample, write RMF3 trajectory + stat file + log -- in one command:

```bash
python run_kcoil_ecoil_sampling.py \
    --copy-number 2 --n-steps 2000 --mode all --output-dir out/
```

See `python run_kcoil_ecoil_sampling.py --help` for all options (sampling
mode, proposal scales, burn-in/thinning, RNG seed, output naming). Outputs
land in `--output-dir` as `<run-name>.rmf3`, `<run-name>_stats.csv`, and
`<run-name>.log`.
