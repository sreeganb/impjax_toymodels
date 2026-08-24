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

`generate_distance_restraints.py` measures that structure and writes a
**sparse, crosslink-like** restraint set to `data/distance_constraints.csv`:

```bash
python generate_distance_restraints.py                    # 10 inter + 2 intra per chain
python generate_distance_restraints.py --top-n 20         # denser
python generate_distance_restraints.py --residue-types K  # strict lysine-only
python generate_distance_restraints.py --explicit-copies 4
```

Sparse is the point, and not only for realism: **every restraint is a separate
node in the exported JAX graph**, so the count is a direct cost. Measured on
this system at `copy_number=1`:

| CSV rows | restraint objects | build+compile | per-eval |
|---:|---:|---:|---:|
| 0 | 3 | 0.33 s | 0.046 ms |
| 14 (default) | 17 | 0.42 s | 0.088 ms |
| 350 | 353 | 11.8 s | 0.692 ms |

### How pairs are chosen

Candidates are restricted to residues a crosslinker actually reacts with
(`--residue-types`, default `KS`), ranked by their distance in the reference
structure, and only the closest `--top-n` are kept. NHS-ester crosslinkers
(DSS, BS3) target lysine, and both proteins are rich in it — but every lysine
here sits in a structured domain, and the `GGSGGGSGGG` linker has none at all,
so a lysine-only dataset would say nothing whatsoever about the flexible
region. NHS esters also react with serine/threonine/tyrosine hydroxyls, and
each linker carries serines at residues 24 and 28, which is why `S` is in the
default set: a few genuine restraints land on the flexible beads instead of
leaving them data-free. The linker beads are never *repositioned* — their
conformation still has to emerge from sampling.

Excluded: pairs inside one rigid body (their distance cannot change, so the
restraint is a constant added to the score) and same-chain pairs closer than
`--min-seq-sep` residues (already covered by connectivity). Several residues
can share one coarse bead, so pairs collapsing onto the same bead pair are
deduplicated.

### File format

```
residue1,protein1,copy1,residue2,protein2,copy2,distance,force_constant
24,KCOIL,*,24,ECOIL,*,4.940,1.000
```

`copy1`/`copy2` are 0-based PMI copy indexes, or `*` meaning "every copy,
paired copy-for-copy" — so one file works unchanged at any `--copy-number`,
which is the default the generator writes. Use `--explicit-copies N` for
integer indexes instead.

Note this makes the synthetic data **more informative than a real experiment**.
Real crosslinking MS is ambiguous about copies — the spectrum says "a KCOIL
K20 crosslinked to an ECOIL S28", not which copy — and the correct treatment
is a soft-min over copy assignments. Assigning each crosslink to a copy by
fiat is what keeps every restraint a cheap two-particle harmonic. Genuinely
ambiguous restraints would need a different restraint class and would change
what `*` means (from "each copy" to "any copy").

`kcoil_ecoil_system.py` reads the file back in and turns each row into a true
harmonic well at `distance` with force constant `force_constant`
(`impjax_toymodels.distance_restraints`), so the restraints are *data*, not
code. On the split prior/likelihood path they sit on the likelihood side,
alongside excluded volume, since they stand in for experimental data;
connectivity remains the prior.

## Benchmark sweep and PDF report

`benchmark_pipeline.py` is a standalone harness over everything above: it
sweeps copy numbers and samplers, measures what came out, and writes a single
multi-page PDF.

```bash
python benchmark_pipeline.py --config data/benchmark_config.json
python benchmark_pipeline.py --config data/benchmark_config.json --skip-run   # re-render only
```

Everything is declared in the JSON — copy numbers, samplers, restraint
selection, proposal scales, per-sampler step counts, and which ground-truth
structure and contact map describe each copy number:

```json
"ground_truth": {
  "structure":   "data/pdb/kcoil_ecoil.pdb",
  "contact_map": "data/contact_maps/kcoil_ecoil_n1.csv",
  "chains":      {"A": ["KCOIL", 0], "B": ["ECOIL", 0]},
  "per_copy_number": {}
}
```

`per_copy_number` is where a genuine N-copy structure goes — `{"2": {...}}`
overrides the default for that case, which is the path a predicted multi-copy
model takes. Without an override the single reference assembly is replicated,
which is what `wildcard_copies` in the restraint block expresses.

Each case gets a **freshly generated restraint set** from its contact map, so
restraint count is a controlled variable rather than whatever was left in
`data/`.

### What the report contains

| Page | |
|---|---|
| Summary | config, JAX backend, case count, and what the numbers do and don't mean |
| Accuracy | RMSD to ground truth, every frame in the scoring window, per copy number |
| IMP score | the same frames re-scored by **IMP on the CPU** |
| Satisfaction | fraction of restraints within tolerance of their target distance |
| Scaling | degrees of freedom against wall and CPU time |
| All results | every number as a table |

The score pages are **IMP's own score, not the BlackJAX log-posterior.** The
latter is `-S(theta) + log p0(theta)` — a different quantity, and quoting it
would make BlackJAX and IMP samplers incomparable. Frames are reloaded into an
IMP model and rescored, which costs ~0.6 ms/frame, so the window is free.

Restraint satisfaction is reported alongside the score because an aggregate
score can hide two badly violated restraints under a hundred satisfied ones.
The deviation is recovered from each restraint's own score: the wells are
`0.5*kappa*(d-d0)^2`, so `|d-d0| = sqrt(2*score/kappa)`.

Every point is one frame, never a mean — for a sampler that returns an
ensemble the spread *is* the result, and averaging it away hides the
difference between a converged run and a wandering one.

Colour identifies the sampler in a fixed order and is stable across every
page. The hues are the first five slots of a palette validated for
colour-vision deficiency (worst adjacent CVD dE 9.1, normal-vision dE 19.6,
against targets of 8 and 15); three sit below 3:1 contrast on white, so every
sampler is also named on the axis or directly labelled, carries its own marker
shape, and the whole sweep is reproduced as a table.

## Did it recover the ground truth?

`evaluate_recovery.py` answers that directly, by walking an RMF3 trajectory
and reporting the RMSD to the reference structure after optimal superposition
(superposed, because every restraint is a function of internal geometry only,
so the recovered assembly is free to sit anywhere in space):

```bash
python evaluate_recovery.py out/kcoil_ecoil_smc_adaptive.rmf3
python evaluate_recovery.py out/*.rmf3 --copy-number 2
```

Only the structured beads are scored: the linker is genuinely flexible and has
no reference conformation to be right or wrong about. With more than one copy
each copy is scored separately against the single reference dimer and the
worst is reported — the restraint file ties copy i to copy i and says nothing
across copies, so the copies are free to land anywhere relative to each other.

`build_kcoil_ecoil_system(copy_number=1, shuffle=False)` gives you the
reference state directly, if you want to check that the ground truth really is
the score minimum.

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
