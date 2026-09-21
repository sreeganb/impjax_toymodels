# examples

Example systems and the shared harness that runs them.

## Layout

```
examples/
  harness/                  system-agnostic: names no molecules
    system_registry.py        --system NAME -> a system module
    generate_contact_map.py
    generate_distance_restraints.py
    run_sampling_comparison.py
    run_imp_rex.py            IMP replica exchange, one replica per MPI rank
    structure_rmsd.py         RMSD to the unshuffled build
    trajectory_analysis.py    best-scoring models, their RMSD, convergence
    benchmark_pipeline.py / benchmark_report.py
  kcoil_ecoil_system.py     system module: the coiled-coil dimer
  rigid_body_docking/       system module + data: 1AVX trypsin/inhibitor
```

Unlike `src/impjax_toymodels/`, code here is not held to the package's
200-300 line/file budget -- it exists to show particular systems going
through the (system-agnostic) wrapper.

`harness/` is shared by every system and mentions none of them. A **system
module** is what tells it what to build; the protocol is documented in
`harness/system_registry.py` and is six names:

```python
PROTEINS, DATA_DIR, DEFAULT_DISTANCE_CSV
build_system(copy_number, data_dir, shuffle, distance_csv)
build_split (copy_number, data_dir, shuffle, distance_csv)
domains_for(protein)
```

Every harness script takes `--system NAME` (default `kcoil_ecoil_system`), and
every benchmark config takes a `"system"` key. Adding a third system means
writing one module and one config, not forking the harness.

Two systems ship:

| | |
|---|---|
| `kcoil_ecoil_system` | coiled-coil dimer: rigid domains joined by a flexible linker. Mostly a question about the linker. |
| `rigid_body_docking/docking_system` | 1AVX trypsin/inhibitor: two rigid bodies, no beads, nothing connecting them. A docking search. See its own [README](rigid_body_docking/README.md). |

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

`harness/generate_distance_restraints.py` measures that structure and writes a
**sparse, crosslink-like** restraint set to `data/distance_constraints.csv`:

```bash
python harness/generate_distance_restraints.py                    # 10 inter + 2 intra per chain
python harness/generate_distance_restraints.py --top-n 20         # denser
python harness/generate_distance_restraints.py --residue-types K  # strict lysine-only
python harness/generate_distance_restraints.py --explicit-copies 4
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

`harness/benchmark_pipeline.py` is a standalone harness over everything above: it
sweeps copy numbers and samplers, measures what came out, and writes a single
multi-page PDF.

```bash
python harness/benchmark_pipeline.py --config data/benchmark_config.json
python harness/benchmark_pipeline.py --config data/benchmark_config.json --skip-run   # re-render only
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

A `"seeds": [0, 1, 2, ...]` list runs the whole sweep once per seed, and each
seed is a *different starting structure* — `seed_imp` seeds IMP's own RNG,
which is what `shuffle_configuration` draws from. On a multimodal target one
seed measures luck as much as sampler quality, so prefer several.

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
| Summary | per sampler: seeds solved, time to solution, best-model and best-50 RMSD, wall/CPU time |
| Time to solution | wall seconds until the run's best-scoring model is within `success_rmsd` of the ground truth |
| Score convergence | best IMP score so far against wall time, ground-truth score as the target |
| Best-scoring models | RMSD of each run's `n_best_models` lowest-scoring models |
| All runs | every number as a table |

**Ground truth** is the system as built, before the shuffle: same
representation, same copy number, rigid bodies on their input coordinates
(`structure_rmsd.py`). One superposition over every rigid-body bead gives one
RMSD per model; flexible beads have no input structure and are left out.

**Best-scoring models.** Every frame of every RMF3 file a run wrote (all
replicas for `imp_rex`; the anneal plus final population for SMC) is
re-scored by **IMP on the CPU** with the full scoring function — never the
BlackJAX log-posterior, which is a different quantity — and the
`n_best_models` lowest after `burnin_fraction` are kept
(`trajectory_analysis.py`, which also runs standalone on any RMF3 files).

**Replica exchange** runs `imp_rex_replicas` replicas under
`imp_rex_launcher` (`mpirun -np N python harness/run_imp_rex.py ...`); load
your MPI/IMP modules first. With 1 replica it runs in-process.

Colour identifies the sampler in a fixed order and is stable across every
page (a CVD-validated palette); every sampler also has its own marker shape
and is named in the legend and the tables.

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

`run_kcoil_ecoil_sampling.py` is a thin forwarder kept so these commands keep
working; the runner itself is `harness/run_sampling_comparison.py`, which takes
`--system` and serves every system.

See `python run_kcoil_ecoil_sampling.py --help` for all options (sampling
mode, proposal scales, SMC particle/temperature-step counts, IMP replica-
exchange frame counts, burn-in/thinning, RNG seed, output naming). Outputs
land in `--output-dir`, named by `--run-name` (default `kcoil_ecoil`):
`<run-name>_rmh.rmf3`/`_stats.csv`, `<run-name>_smc.rmf3`/`_stats.csv`
(best-scoring particle per temperature step only), `<run-name>_imp_rex/`
(IMP's own output directory), and one shared `<run-name>.log` for whichever
samplers were run.
