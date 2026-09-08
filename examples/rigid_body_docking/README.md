# rigid_body_docking

A second system for the `examples/harness/` pipeline: **1AVX**, porcine
pancreatic trypsin bound to soybean trypsin inhibitor. Two rigid bodies, no
flexible beads, and nothing connecting them — a genuine protein–protein
docking search rather than the linker-refinement question the KCOIL/ECOIL
example asks.

Everything in `harness/` is shared and system-agnostic. Only two files here
know what the molecules are: `docking_system.py` (how to build it) and
`docking_metrics.py` (how to score a docking model). Adding a third system
means writing those two and a config, not forking the harness.

| | |
|---|---|
| Rigid bodies | 2 (TRYP 112 beads, STI 86 beads, resolution 2) |
| Flexible beads | 0 |
| Sampled parameters | 14 (quaternion + translation per body) |
| True degrees of freedom | 12, of which only the 6 relative ones change the score |
| Restraints | 20 synthetic interface crosslinks + excluded volume |

## Preparing the structure

```bash
python prepare_1avx.py                 # fetch from RCSB, observed residues only
python prepare_1avx.py --pdb 1AVX.pdb  # use a local copy
python prepare_1avx.py --model-gaps    # bead the 5 unobserved residues instead
```

The deposited entry cannot go straight into PMI. Trypsin is numbered by
alignment to chymotrypsinogen, so three residues carry insertion codes
(184A, 188A, 221A). IMP's PDB reader keeps the residue index and drops the
code, so it reads **224 residues with 3 duplicate indexes**; PMI's
`add_structure` keys on that index, the duplicates collide, and the built
system is quietly wrong with no error raised anywhere.

The same numbering scheme puts six apparent gaps in chain A (34→37, 67→69,
125→127, 130→132, 204→209, 217→219). **None of them are missing residues** —
every C–N bond across them is under 1.45 Å, so the chain is chemically
continuous; they are deletions relative to chymotrypsinogen. Reading them as
gaps would insert ten residues that are physically present, which is why
breaks are detected from the peptide bond rather than from the numbering.

Chain B has exactly one genuine break: residues 639–643 (QAEDD), with
C638–N644 spanning 11.41 Å. That is the only unmodelled region in the whole
complex — 5 residues out of 400.

So both chains are renumbered contiguously from 1, each FASTA is derived from
the residues actually present, and `data/residue_mapping.csv` records the
original numbering so any restraint can be traced back to the deposited file.
By default the 5-residue gap is spliced out, which is what makes every residue
structured and the system exactly two rigid bodies. `--model-gaps` instead
allocates them from `REMARK 465`, giving STI domains `[[1,138],[144,177]]`
against the full 177-residue SEQRES, and the mixed rigid/flexible path.

## The synthetic data

```bash
python ../harness/generate_contact_map.py --system docking_system \
    --structure data/pdb/1avx_complex.pdb --chains A=TRYP:0 B=STI:0 \
    --output data/contact_maps/1avx_n1.csv

python ../harness/generate_distance_restraints.py --system docking_system \
    --contact-map data/contact_maps/1avx_n1.csv \
    --structure data/pdb/1avx_complex.pdb --chains A=TRYP:0 B=STI:0 \
    --top-n 20 --residue-types KS --wildcard-copies \
    --output data/distance_constraints.csv
```

20 restraints, all inter-molecular, on lysine and serine — the residues an
NHS-ester crosslinker (DSS, BS3) actually reacts with. Their residue-level
distances run 10–16 Å, comfortably inside a real DSS span.

`--top-n-intra` does nothing here and does not need setting: with one rigid
body per chain, every intra-chain pair has both ends in the same body, and
`contact_map.select` already drops those because their distance cannot change
— a restraint on one only adds a constant to the score.

The crystal pose sits at the minimum of these restraints to **5e-4 Å** (bounded
by the CSV's 3-decimal targets), so "recovered the structure" and "lowered the
score" really are the same question here. A shuffled start scores 65661, with
the worst restraint 88 Å from its target.

## Running the samplers

```bash
python ../harness/run_sampling_comparison.py --system docking_system \
    --samplers rmh smc smc_tempered smc_adaptive imp_rex \
    --mode rigid --prior split+box --prior-box-half-width 100 \
    --sigma-rotation 0.02 --sigma-translation 0.25 \
    --output-dir out/ --run-name 1avx
```

`--mode rigid` is equivalent to `all` here (there are no beads), but saying so
explicitly documents the intent.

**Proposal scales are tuned for this system, not inherited.** The KCOIL/ECOIL
defaults (σ_rot 0.05, σ_trans 1.0) give 12.7% acceptance on these much larger
bodies. A sweep over σ_rot ∈ [0.005, 0.4] × σ_trans ∈ [0.15, 8.0] put
**σ_rot 0.02, σ_trans 0.25 at 30.9%**, which is what the config uses.

### The prior/likelihood split

```
λ-tempered likelihood : Σ 0.5·κ·(d − d₀)²      the 20 interface crosslinks
untempered prior      : excluded volume  +  bounding box (half-width 100 Å)
```

This inverts the KCOIL/ECOIL arrangement deliberately. Excluded volume is
structural prior knowledge about proteins, not an experimental observation.
It also fixes the failure mode recorded there: excluded volume alone has
almost no variance across draws from a box prior — scattered bodies never
clash — so tempering it was close to a no-op. The interface term goes from 0
at the crystal pose to 65661 after a shuffle, so the anneal has real work.

Connectivity is absent unless the representation has flexible beads. With one
rigid body per chain it evaluates to a constant, and every restraint is a node
in the exported JAX graph.

## Measured: the JAX excluded-volume score carries a constant offset

IMP has no JAX path for close-pair containers and says so (`No JAX support for
close pairs, so using all possible pairs`). On the CPU, `ExcludedVolumeSphere`
filters out pairs of beads sharing a rigid body; the all-pairs JAX export
cannot, and consecutive coarse beads overlap by construction — a resolution-2
bead has radius 2.9–4.2 Å and its neighbour's centre is ~5 Å away. At the
crystal pose:

| | |
|---:|---|
| 466.1924 | all-pairs total — what `score_func` returns |
| 455.6515 | intra-body, 374 overlapping pairs, both ends in one rigid body |
| 10.5408 | inter-body, 7 pairs — **equals IMP's CPU score exactly** |

The intra-body part is an **exact additive constant** in the sampled
posterior: those pairs are rigid-body members, so their mutual distances
cannot change. Measured at 455.6515 ± 2e-4 (float32 rounding in the quaternion
rotation) across rigid moves that swing the inter-body term from 0 to 56.6.

A constant in the log-density cancels in every Metropolis acceptance ratio,
and — since excluded volume sits in the untempered prior here — in SMC's
incremental weights too. **Sampling is unaffected**; only the absolute number
differs from IMP's, which is why the benchmark re-scores frames with IMP on
the CPU instead of quoting the BlackJAX log-posterior. Expect `--debug` score
verification to report this offset; it is understood, not a bug in the
expansion map.

## Measuring accuracy

`harness/evaluate_recovery.py` gives the global superposed RMSD over every
structured bead, which keeps this comparable with the KCOIL/ECOIL report. On
its own it is the wrong primary question for docking: a global superposition
spreads the error across both partners. Translating the ligand 5 Å gives a
global RMSD of **1.50** but a ligand RMSD of **5.00**.

`docking_metrics.py` adds the three measurements the field actually uses:

| | |
|---|---|
| **L-RMSD** | superpose the receptor, measure the ligand — all the error lands on the relative placement |
| **I-RMSD** | interface beads only; insensitive to the lever arm that inflates L-RMSD when a distant part of the ligand swings |
| **fnat** | fraction of native contacts recovered; needs no superposition at all |
| **CAPRI class** | high / medium / acceptable / incorrect |

Cutoffs are derived from this representation rather than copied from the
literature. CAPRI defines contacts between heavy atoms at 5 Å; a resolution-2
bead has a mean radius of 3.6 Å, so bead-centre distances run about two mean
radii longer, putting the equivalent cutoff near **12 Å**. At that cutoff the
crystal pose has 74 native bead contacts across 47 interface beads. The CAPRI
class is an *indication* only — the published thresholds were calibrated on
all-atom RMSDs, and a coarse-grained model cannot be held to a 1 Å criterion.

### On PMI_analysis

`PMI_analysis/pyext/src/accuracy.py` is a wrapper over
`IMP.pmi.analysis.Precision.get_rmsd_wrt_reference_structure_with_alignment`,
and it needs `cluster.N.sample_A/B.txt` files produced by the full
`run_analysis_trajectories → extract_models → run_clustering` workflow, which
is built around replica-exchange output directories rather than the RMF3
trajectories the BlackJAX samplers write. (Its own `example/get_accuracy.py`
also calls `AccuracyModels` with keyword arguments that no longer match the
class.)

So the underlying primitive is used directly instead of the pipeline around
it. `test/test_docking_metrics.py` writes real RMF3 files and asserts our
Kabsch RMSD matches `IMP.pmi.analysis.Precision` — measured agreement
**1.7e-6 Å** on shuffled frames ~30 Å from the reference.

## The benchmark

```bash
python ../harness/benchmark_pipeline.py --config data/benchmark_config.json
python ../harness/benchmark_pipeline.py --config data/benchmark_config.json --skip-run
```

All five samplers on the same system and the same scoring function: BlackJAX
RMH, three SMC variants, and IMP's own replica exchange as the baseline. The
report adds three docking pages (L-RMSD, I-RMSD, fnat) to the standard ones,
driven by the config's `metrics_module` key; a system without one gets the
standard report unchanged.

### Seeds are not optional here

The config runs the sweep over `"seeds": [0, 1, 2, 3, 4]`, and each seed is a
**different starting structure** — `seed_imp` seeds IMP's own RNG, which is
what `shuffle_configuration` draws from. That matters because two rigid bodies
with nothing tethering them is a genuinely multimodal search: before seeding
was fixed, two runs of this exact config measured the same sampler docking to
1.6 Å and failing at 17.6 Å. A single seed here measures luck as much as
sampler quality.

Seeding also makes the comparison fair. Every sampler builds its own system
(sampling mutates the IMP model in place, so they cannot share one), which
previously meant each got a *different* random start.

## Results

Measured on this machine (JAX on CPU), 5 seeds, wall time equalised at
~6–8 s per sampler — `imp_rex` is at 6000 frames for that reason; at 2000 it
finished in 3 s and any reliability claim would have been unfair.

Final-window median RMSD to the crystal structure, per seed:

| sampler | s0 | s1 | s2 | s3 | s4 | docked | wall s | cpu s |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|
| `rmh`          |  1.46 |  1.07 | 16.07 |  1.07 |  1.52 | 4/5 | 6.2 |  8.9 |
| `smc`          | 16.10 |  0.91 |  0.93 |  0.94 |  1.01 | 4/5 | 7.9 | 24.3 |
| `smc_tempered` | 16.03 |  0.92 |  0.88 |  0.86 |  0.83 | 4/5 | 7.6 | 25.4 |
| `smc_adaptive` |  1.00 |  1.02 |  1.10 |  1.14 |  1.61 | **5/5** | 8.3 | 26.3 |
| `imp_rex`      |  1.05 |  9.67 | 17.23 | 17.18 |  1.19 | 2/5 | 7.2 |  7.1 |

Best frame found across all seeds:

| sampler | RMSD | L-RMSD | I-RMSD | fnat | CAPRI |
|---|---:|---:|---:|---:|:---:|
| `rmh`          | 0.57 | 1.05 | 0.47 | 0.91 | high |
| `smc`          | 0.65 | 1.37 | 0.54 | 0.76 | high |
| `smc_tempered` | 0.66 | 1.42 | 0.53 | 0.77 | high |
| `smc_adaptive` | 0.67 | 1.29 | 0.56 | 0.84 | high |
| `imp_rex`      | 0.36 | 1.08 | 0.31 | 1.00 | high |

Read together these say something the aggregate score alone would not:

* **Every sampler can solve this problem.** All five reach CAPRI *high*
  quality on their best frame, sub-ångström in RMSD. The 20 synthetic
  crosslinks determine the 6 relative degrees of freedom well.
* **They differ in reliability, not in ceiling.** Adaptive-tempered SMC
  docked from all five starts; RMH and the fixed-ladder SMC variants from
  four; IMP's replica exchange from two, at comparable wall time. Adapting
  the temperature ladder to hold ESS is doing real work on a multimodal
  target — which is the case the fixed ladder was never going to cover, since
  its schedule cannot notice that the population has collapsed.
* **When replica exchange does find the interface it refines best** (0.36 Å,
  every native contact recovered) — unsurprising for a well-tuned MC with
  adaptive movers. Its problem here is search, not refinement.
* **Wall time hides a large CPU-time difference.** The SMC variants use
  ~25 s of CPU for ~8 s of wall time; RMH uses 8.9 s and replica exchange
  7.1 s. The SMC population vectorises, so it is buying reliability with
  parallel work — which is exactly the trade a GPU should make cheaper, and
  the number to re-measure on one.

These are single-machine, CPU-backend numbers on one 14-parameter system, not
a general claim about the samplers.
