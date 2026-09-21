"""Turn a benchmark sweep's records into a short multi-page PDF report.

Split out from benchmark_pipeline.py so a report can be re-rendered from a
finished results.json without re-running anything (`--skip-run`).

Pages, in reading order
-----------------------
1. **Summary** -- one row per sampler: how many seeds it solved, how long that
   took, and how accurate its best-scoring models are. The whole argument on
   one page.
2. **Time to solution** -- wall seconds until a run's best-scoring model is
   within `success_rmsd` of the ground truth, one point per seed. This is the
   comparison that folds in both sampling efficiency and hardware: an SMC
   population evaluated in parallel on a GPU shortens it directly, a replica
   ladder spread over N CPU ranks shortens it only as far as N allows.
3. **Score convergence** -- best IMP score found so far against wall time, one
   line per seed, with the ground-truth score as the target line.
4. **RMSD of the best-scoring models** -- every one of each run's
   `n_best_models` lowest-scoring models, seeds pooled.
5. **All runs** -- every number as text.

Every score is IMP's, re-evaluated on the CPU with the full scoring function,
never a sampler's own log-posterior, so all samplers sit on one scale.

Chart conventions
-----------------
Colour identifies the sampler, in a fixed order, on every page -- it follows
the entity, never its rank. The five hues are the first five slots of a
palette validated for colour-vision deficiency (worst adjacent CVD deltaE 9.1,
worst normal-vision deltaE 19.6, OKLab x100). Three of them sit below 3:1
contrast on white, so every sampler also has its own marker shape, is named on
the axis or in the legend, and appears in the text table.
"""

import json
import os
from typing import List, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

#: Fixed sampler -> (colour, marker) assignment.
SERIES_STYLE = {
    "rmh":          ("#2a78d6", "o"),
    "smc":          ("#eb6834", "s"),
    "smc_tempered": ("#1baf7a", "^"),
    "smc_adaptive": ("#eda100", "D"),
    "imp_rex":      ("#e87ba4", "v"),
}
FALLBACK_STYLE = ("#4a3aa7", "P")

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"

#: Per-run arrays that must survive the JSON round trip for --skip-run.
ARRAY_FIELDS = ("best_scores", "best_rmsds", "trace_time", "trace_score", "trace_rmsd")


def style_of(sampler: str):
    return SERIES_STYLE.get(sampler, FALLBACK_STYLE)


def save_records(path: str, records: Sequence[dict]) -> None:
    """Persist a sweep, converting numpy arrays to lists."""
    serialisable = []
    for record in records:
        row = dict(record)
        for field in ARRAY_FIELDS:
            if row.get(field) is not None:
                row[field] = np.asarray(row[field]).tolist()
        serialisable.append(row)
    with open(path, "w") as handle:
        json.dump(serialisable, handle, indent=2)


def load_records(path: str) -> List[dict]:
    with open(path) as handle:
        records = json.load(handle)
    for record in records:
        for field in ARRAY_FIELDS:
            if record.get(field) is not None:
                record[field] = np.asarray(record[field], dtype=float)
    return records


# --------------------------------------------------------------------------- helpers

def _samplers(records) -> List[str]:
    """Samplers present, in the palette's fixed order."""
    present = {r["sampler"] for r in records}
    ordered = [s for s in SERIES_STYLE if s in present]
    return ordered + sorted(present - set(ordered))


def _runs(records, copy_number, sampler) -> List[dict]:
    return [r for r in records if r["copy_number"] == copy_number
            and r["sampler"] == sampler and not r.get("failure")]


def _median(values) -> float:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def _fmt(value, spec=".1f", missing="-") -> str:
    return missing if value is None or not np.isfinite(value) else format(value, spec)


def _frame(axes, ylabel: str = "", title: str = "") -> None:
    """Recessive grid and axes, so the marks carry the chart."""
    axes.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    if title:
        axes.set_title(title, color=INK, fontsize=10, pad=8)
    axes.grid(axis="y", color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_SOFT, labelsize=8)


def _legend(figure, samplers, extra=()) -> None:
    handles = [Line2D([], [], color=style_of(s)[0], marker=style_of(s)[1],
                      markersize=7, linewidth=2, label=s) for s in samplers]
    figure.legend(handles=handles + list(extra), loc="lower center",
                  ncol=min(len(handles) + len(extra), 6), frameon=False,
                  fontsize=8, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, 0.005))


def _page(records, title, subtitle):
    """A figure with one panel per copy number, titled."""
    copy_numbers = sorted({r["copy_number"] for r in records})
    figure, axes = plt.subplots(1, len(copy_numbers), figsize=(11, 5.4), squeeze=False)
    figure.text(0.5, 0.955, title, ha="center", fontsize=14, color=INK)
    figure.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=8.5, color=INK_SOFT)
    return figure, list(zip(axes[0], copy_numbers))


def _finish(pdf, figure) -> None:
    figure.tight_layout(rect=[0, 0.06, 1, 0.84])
    pdf.savefig(figure)
    plt.close(figure)


def _category_axis(axis, samplers) -> None:
    axis.set_xticks(range(len(samplers)))
    axis.set_xticklabels([s.replace("smc_", "smc\n") for s in samplers], fontsize=7.5)
    axis.set_xlim(-0.7, len(samplers) - 0.3)


def _jitter(n, seed) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) - 0.5) * 0.3


# --------------------------------------------------------------------------- pages

def _summary_rows(config, records):
    rows = []
    for copy_number in sorted({r["copy_number"] for r in records}):
        for sampler in _samplers(records):
            runs = _runs(records, copy_number, sampler)
            if not runs:
                continue
            solved = [r["time_to_solution"] for r in runs if np.isfinite(r["time_to_solution"])]
            best_model = [r["best_rmsds"][0] for r in runs if len(r["best_rmsds"])]
            rows.append([
                str(copy_number), sampler,
                f"{len(solved)}/{len(runs)}",
                _fmt(_median(solved)),
                _fmt(_median(best_model), ".2f"),
                _fmt(_median([_median(r["best_rmsds"]) for r in runs]), ".2f"),
                _fmt(_median([r["wall_time"] for r in runs])),
                _fmt(_median([r["cpu_time"] for r in runs])),
            ])
    return rows


def _table(figure, rect, header, rows, widths, sampler_col=1):
    axis = figure.add_axes(rect)
    axis.axis("off")
    rendered = axis.table(cellText=rows, colLabels=header, loc="upper center",
                          cellLoc="center", colWidths=widths)
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8)
    rendered.scale(1, 1.5)
    for (row, col), cell in rendered.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(color=INK, fontweight="bold")
        else:
            cell.set_text_props(color=INK_SOFT)
            if col == sampler_col:  # colour chip beside the name, never instead of it
                cell.get_text().set_color(style_of(rows[row - 1][sampler_col])[0])


def _summary_page(pdf, config, records):
    figure = plt.figure(figsize=(11, 8.5))
    figure.text(0.06, 0.93, config.get("name", "benchmark"), fontsize=22, color=INK)
    figure.text(0.06, 0.895, "IMP + JAX/BlackJAX sampler benchmark", fontsize=12, color=INK_SOFT)

    backends = sorted({r.get("backend", "?") for r in records})
    seeds = sorted({r["seed"] for r in records})
    replicas = config.get("sampler_params", {}).get("imp_rex_replicas", 1)
    lines = [
        f"system {config.get('system', '?')}   |   copy numbers {config['copy_numbers']}   |   "
        f"seeds {seeds}   |   JAX backend {', '.join(backends)}   |   imp_rex replicas {replicas}",
        f"accuracy = RMSD to the unshuffled build, over the {config['n_best_models']} "
        f"best-scoring models after a {100 * config['burnin_fraction']:.0f}% burn-in   |   "
        f"solved = best-scoring model within {config['success_rmsd']} A",
    ]
    for index, line in enumerate(lines):
        figure.text(0.06, 0.84 - 0.03 * index, line, fontsize=9, color=INK_SOFT)

    header = ["copies", "sampler", "seeds solved", "time to solution (s)",
              "best-model RMSD (A)", f"best-{config['n_best_models']} RMSD (A)",
              "wall (s)", "CPU (s)"]
    _table(figure, [0.06, 0.25, 0.88, 0.52], header, _summary_rows(config, records),
           [0.07, 0.14, 0.11, 0.16, 0.15, 0.15, 0.1, 0.1])

    notes = ("Medians over seeds. Time to solution counts only seeds that were solved. "
             "Wall time is the sampler alone; for\nimp_rex it is the slowest replica, "
             "and CPU time is summed over replicas.")
    figure.text(0.06, 0.17, notes, fontsize=8.5, color=INK_SOFT, va="top")
    if backends == ["cpu"]:
        figure.text(0.06, 0.10,
                    "JAX ran on CPU here. The BlackJAX samplers score whole populations in "
                    "one vectorised call, so their wall time is the\nnumber a GPU run changes; "
                    "re-run this config on a GPU host to measure it.",
                    fontsize=8.5, color="#c8372d", va="top")
    pdf.savefig(figure)
    plt.close(figure)


def _time_to_solution_page(pdf, config, records):
    figure, panels = _page(
        records, "Time to solution",
        f"Wall seconds until the run's best-scoring model is within {config['success_rmsd']} A "
        "of the ground truth. One point per seed; lower is better.\n"
        "Seeds that never got there sit in the 'not solved' band at the top.")
    samplers = _samplers(records)
    for axis, copy_number in panels:
        solved_values, unsolved = [], []
        for index, sampler in enumerate(samplers):
            runs = _runs(records, copy_number, sampler)
            times = np.array([r["time_to_solution"] for r in runs], dtype=float)
            ok = times[np.isfinite(times)]
            solved_values.extend(ok)
            unsolved.append((index, int((~np.isfinite(times)).sum())))
            colour, marker = style_of(sampler)
            axis.scatter(index + _jitter(ok.size, index), ok, s=40, color=colour,
                         marker=marker, edgecolors="white", linewidths=0.8, zorder=3)
            if ok.size:
                median = float(np.median(ok))
                axis.plot([index - 0.3, index + 0.3], [median] * 2, color=colour,
                          linewidth=2, solid_capstyle="round", zorder=4)
                axis.annotate(f"{median:.3g}s", (index + 0.33, median), fontsize=7,
                              color=INK_SOFT, va="center")
        top = max(solved_values) * 1.6 if solved_values else 10.0
        bottom = min(solved_values) / 1.6 if solved_values else 1.0
        axis.set_yscale("log")
        axis.set_ylim(bottom, top * 1.6)
        axis.axhspan(top, top * 1.6, color=GRID, alpha=0.35, zorder=0)
        for index, count in unsolved:
            if count:
                axis.text(index, top * 1.25, f"{count} not solved", ha="center",
                          va="center", fontsize=7, color=INK_SOFT)
        _category_axis(axis, samplers)
        _frame(axis, "wall seconds (log)", f"copy number {copy_number}")
    _legend(figure, samplers)
    _finish(pdf, figure)


def _convergence_page(pdf, config, records):
    figure, panels = _page(
        records, "Score convergence",
        "Best IMP score found so far against wall time; one line per seed. "
        "Dashed: the IMP score of the ground truth.\n"
        "A curve that reaches the dashed line sooner has found a structure as good as the "
        "true one sooner.")
    samplers = _samplers(records)
    for axis, copy_number in panels:
        all_scores = []
        for sampler in samplers:
            colour, marker = style_of(sampler)
            for run in _runs(records, copy_number, sampler):
                times, scores = run["trace_time"], run["trace_score"]
                if not len(times):
                    continue
                # Extend the last value to the end of the run: it is still the best.
                xs = np.append(times, max(run["wall_time"], times[-1]))
                ys = np.append(scores, scores[-1])
                axis.step(xs, ys, where="post", color=colour, linewidth=1.4, alpha=0.8)
                axis.plot(xs[-1], ys[-1], marker=marker, color=colour, markersize=6,
                          markeredgecolor="white")
                all_scores.extend(scores)
        reference = [r["reference_score"] for r in records if r["copy_number"] == copy_number]
        if reference:
            axis.axhline(reference[0], color=INK_SOFT, linewidth=1, linestyle="--")
            all_scores.append(reference[0])
        positive = [s for s in all_scores if s > 0]
        axis.set_xscale("log")
        if positive and len(positive) == len(all_scores) and max(positive) / min(positive) > 10:
            axis.set_yscale("log")
        axis.set_xlabel("wall seconds (log)", color=INK_SOFT, fontsize=9)
        _frame(axis, "best IMP score so far", f"copy number {copy_number}")
    target = Line2D([], [], color=INK_SOFT, linestyle="--", linewidth=1,
                    label="ground-truth score")
    _legend(figure, samplers, extra=[target])
    _finish(pdf, figure)


def _best_models_page(pdf, config, records):
    n_best = config["n_best_models"]
    figure, panels = _page(
        records, f"RMSD of the {n_best} best-scoring models",
        f"Each run's {n_best} lowest IMP-score models after a "
        f"{100 * config['burnin_fraction']:.0f}% burn-in, all seeds pooled; RMSD to the "
        "unshuffled build,\none superposition over every rigid-body bead. The rule is the "
        f"median; dashed is the {config['success_rmsd']} A success threshold. Lower is better.")
    samplers = _samplers(records)
    for axis, copy_number in panels:
        for index, sampler in enumerate(samplers):
            runs = _runs(records, copy_number, sampler)
            values = np.concatenate([r["best_rmsds"] for r in runs]) if runs else np.array([])
            if not values.size:
                continue
            colour, marker = style_of(sampler)
            axis.scatter(index + _jitter(values.size, index), values, s=12, color=colour,
                         marker=marker, alpha=0.5, edgecolors="white", linewidths=0.5,
                         zorder=3)
            median = float(np.median(values))
            axis.plot([index - 0.3, index + 0.3], [median] * 2, color=colour,
                      linewidth=2.2, solid_capstyle="round", zorder=4)
            axis.annotate(f"{median:.2f}", (index + 0.33, median), fontsize=7,
                          color=INK_SOFT, va="center")
        axis.axhline(config["success_rmsd"], color=INK_SOFT, linewidth=1, linestyle="--")
        axis.set_ylim(bottom=0)
        _category_axis(axis, samplers)
        _frame(axis, "RMSD to ground truth (A)", f"copy number {copy_number}")
    _legend(figure, samplers)
    _finish(pdf, figure)


def _table_page(pdf, config, records):
    """Every run as text: the colour-independent route to all of the data."""
    figure = plt.figure(figsize=(11, 8.5))
    figure.suptitle("All runs", fontsize=13, color=INK, y=0.96)
    header = ["copies", "sampler", "seed", "DOF", "wall s", "CPU s", "frames",
              "solved at s", "best-model RMSD", "median RMSD", "best IMP score", "note"]
    rows = []
    for r in sorted(records, key=lambda r: (r["copy_number"], r["seed"], r["sampler"])):
        rmsds = r.get("best_rmsds")
        scores = r.get("best_scores")
        has = rmsds is not None and len(rmsds)
        rows.append([
            str(r["copy_number"]), r["sampler"], str(r["seed"]), str(r.get("n_dof", "-")),
            _fmt(r["wall_time"]), _fmt(r.get("cpu_time", float("nan"))),
            str(r.get("n_frames", "-")), _fmt(r.get("time_to_solution", float("nan"))),
            _fmt(rmsds[0], ".2f") if has else "-",
            _fmt(float(np.median(rmsds)), ".2f") if has else "-",
            _fmt(scores[0], ".2f") if has else "-",
            (r.get("failure") or r.get("note") or "")[:24],
        ])
    # One page holds ~40 rows; a longer sweep is split across pages.
    for start in range(0, max(len(rows), 1), 40):
        if start:
            figure = plt.figure(figsize=(11, 8.5))
            figure.suptitle("All runs (continued)", fontsize=13, color=INK, y=0.96)
        _table(figure, [0.03, 0.03, 0.94, 0.88], header, rows[start:start + 40],
               [0.05, 0.1, 0.04, 0.05, 0.06, 0.06, 0.06, 0.08, 0.1, 0.09, 0.09, 0.18])
        pdf.savefig(figure)
        plt.close(figure)


def write_report(path: str, config: dict, records: Sequence[dict]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Report-level defaults, so an older results.json still renders.
    config = {"n_best_models": 50, "burnin_fraction": 0.25, "success_rmsd": 5.0, **config}
    with PdfPages(path) as pdf:
        _summary_page(pdf, config, records)
        _time_to_solution_page(pdf, config, records)
        _convergence_page(pdf, config, records)
        _best_models_page(pdf, config, records)
        _table_page(pdf, config, records)
    return path
