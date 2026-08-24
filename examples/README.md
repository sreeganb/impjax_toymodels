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

## Distance restraints derived from the ground truth

`examples/data/pdb/kcoil_ecoil.pdb` is the reference complex this toy study
is trying to recover: chain A is exactly `KCOIL.pdb` and chain B is exactly
`ECOIL.pdb`, so the two per-protein files are already in a common frame and
"the ground truth" is simply the built system *before* `shuffle_configuration`
scrambles it.

`generate_distance_restraints.py` measures that structure and writes every
informative contact to `data/distance_constraints.csv`:

```bash
python generate_distance_restraints.py                  # defaults: 12 A cutoff
python generate_distance_restraints.py --cutoff 8 --force-constant 2.0
python generate_distance_restraints.py --explicit-copies 4
```

The file has one restraint per row:

```
residue1,protein1,copy1,residue2,protein2,copy2,distance,force_constant
25,KCOIL,*,24,ECOIL,*,4.570,1.000
```

`copy1`/`copy2` are 0-based PMI copy indexes, or `*` meaning "every copy,
paired copy-for-copy" -- so one file works unchanged at any `--copy-number`,
which is the default the generator writes.  Use `--explicit-copies N` to get
integer copy indexes instead, for restraint sets where different copies of the
assembly genuinely differ.

`kcoil_ecoil_system.py` reads the file back in and turns each row into a true
harmonic well at `distance` with force constant `force_constant`
(`impjax_toymodels.distance_restraints`), so the restraints are *data*, not
code: a sparser, noisier or differently-weighted set is a different CSV, not a
different build script.  On the split prior/likelihood path they sit on the
likelihood side, alongside excluded volume, since they stand in for
experimental data; connectivity remains the prior.

Pairs inside one rigid body are skipped (their distance cannot change) as are
same-chain pairs closer than `--min-seq-sep` residues (connectivity already
covers those).  The generator prints how many restraints tie each pair of
rigid bodies together, and flags any pair with fewer than the three needed to
fix a relative pose.

To put a built system back into the reference state -- for a sanity check that
the ground truth really is the score minimum, or to compare a sampled model
against it:

```python
built, sf, _ = build_kcoil_ecoil_system(copy_number=1, shuffle=False)
place_flexible_beads_at_reference(built)   # linkers have no PDB-derived start
```

## Did it recover the ground truth?

`evaluate_recovery.py` answers that directly, by walking an RMF3 trajectory
and reporting the RMSD to the reference structure after optimal superposition
(superposed, because every restraint is a function of internal geometry only,
so the recovered assembly is free to sit anywhere in space):

```bash
python evaluate_recovery.py out/kcoil_ecoil_smc_adaptive.rmf3
python evaluate_recovery.py out/*.rmf3 --copy-number 2
```

With more than one copy each copy is scored separately against the single
reference dimer and the worst is reported -- the restraint file ties copy i to
copy i and says nothing across copies, so the copies are free to land anywhere
relative to each other.

## Running the samplers

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
