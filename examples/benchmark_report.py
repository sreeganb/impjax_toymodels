"""Turn a benchmark sweep's records into a multi-page PDF report.

Split out from benchmark_pipeline.py so a report can be re-rendered from a
finished results.json without re-running anything (`--skip-run`), which is what
you want while iterating on how the results are presented.

Chart conventions
-----------------
Colour identifies the **sampler**, in a fixed order, and the same sampler wears
the same colour on every page -- colour follows the entity, never its rank or
its position in a filtered list.  The five hues are the first five slots of a
palette validated for colour-vision deficiency: worst adjacent CVD deltaE 9.1
and worst normal-vision deltaE 19.6 (OKLab x100, against targets of 8 and 15).

Three of those hues sit below 3:1 contrast on a white page, which obliges
visible labels rather than colour alone -- so every sampler is also named on
the category axis or directly labelled, carries its own marker shape, and the
whole sweep is reproduced as a table on the last page.  Nothing here is
readable only by hue.

Distributions are drawn as the individual frames in the scoring window, not as
a mean: the spread across the last N frames *is* the result for a sampler that
returns an ensemble, and averaging it away would hide exactly the difference
between a converged run and a wandering one.
"""

import json
import os
from typing import Dict, List, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

#: Fixed sampler -> (colour, marker) assignment.  Order is the palette's, and
#: markers are a second, colour-independent channel for print and CVD.
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

#: Arrays that have to survive the JSON round trip for --skip-run.
ARRAY_FIELDS = ("rmsd", "imp_score", "satisfied")


def style_of(sampler: str):
    return SERIES_STYLE.get(sampler, FALLBACK_STYLE)


def save_records(path: str, records: Sequence[dict]) -> None:
    """Persist a sweep, converting numpy arrays to lists."""
    serialisable = []
    for record in records:
        row = dict(record)
        for field in ARRAY_FIELDS:
            if field in row and row[field] is not None:
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
                record[field] = np.asarray(record[field])
    return records


def _frame(axes, ylabel: str, title: str = "") -> None:
    """Recessive grid and axes, so the marks carry the chart."""
    axes.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    if title:
        axes.set_title(title, color=INK, fontsize=10, pad=8)
    axes.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_SOFT, labelsize=8)


def _legend(figure, samplers: Sequence[str]) -> None:
    """A legend is always present for two or more series."""
    handles = [Line2D([], [], color=style_of(s)[0], marker=style_of(s)[1],
                      linestyle="none", markersize=7, label=s)
               for s in samplers]
    figure.legend(handles=handles, loc="lower center", ncol=min(len(samplers), 5),
                  frameon=False, fontsize=8, labelcolor=INK_SOFT,
                  bbox_to_anchor=(0.5, 0.005))


def _window_panels(pdf, records, field, ylabel, title, subtitle, log=False):
    """One panel per copy number; every frame in the window drawn as a point."""
    copy_numbers = sorted({r["copy_number"] for r in records})
    samplers = [s for s in SERIES_STYLE
                if any(r["sampler"] == s for r in records)]
    samplers += [r["sampler"] for r in records if r["sampler"] not in SERIES_STYLE]
    samplers = list(dict.fromkeys(samplers))

    figure, axes_row = plt.subplots(
        1, len(copy_numbers), figsize=(11, 5.2), sharey=True, squeeze=False)
    figure.suptitle(title, fontsize=13, color=INK, x=0.5, y=0.97)
    figure.text(0.5, 0.915, subtitle, ha="center", fontsize=8.5, color=INK_SOFT)

    for axis, copy_number in zip(axes_row[0], copy_numbers):
        positions, labels = [], []
        for index, sampler in enumerate(samplers):
            match = [r for r in records
                     if r["copy_number"] == copy_number and r["sampler"] == sampler]
            positions.append(index)
            labels.append(sampler.replace("smc_", "smc\n"))
            if not match or match[0].get(field) is None:
                continue
            values = np.asarray(match[0][field], dtype=float)
            if not values.size:
                continue
            colour, marker = style_of(sampler)
            jitter = (np.random.default_rng(index).random(values.size) - 0.5) * 0.28
            axis.scatter(index + jitter, values, s=14, color=colour, marker=marker,
                         alpha=0.55, linewidths=0.6, edgecolors="white", zorder=3)
            # Median as a short rule: one direct, unambiguous readout per series.
            median = float(np.median(values))
            axis.plot([index - 0.32, index + 0.32], [median, median],
                      color=colour, linewidth=2.0, solid_capstyle="round", zorder=4)
            axis.annotate(f"{median:.3g}", (index + 0.34, median), fontsize=7,
                          color=INK_SOFT, va="center", ha="left")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=7.5)
        axis.set_xlim(-0.7, len(samplers) - 0.3)
        if log:
            axis.set_yscale("log")
        _frame(axis, ylabel if copy_number == copy_numbers[0] else "",
               f"copy number {copy_number}")

    _legend(figure, samplers)
    figure.tight_layout(rect=[0, 0.06, 1, 0.90])
    pdf.savefig(figure)
    plt.close(figure)


def _autoscale(axis, values, set_scale) -> None:
    """Use a log scale only if the values span more than a decade."""
    finite = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if finite and max(finite) / min(finite) > 10.0:
        set_scale("log")


def _scaling_page(pdf, records):
    """System size against wall time, one line per sampler."""
    samplers = list(dict.fromkeys(r["sampler"] for r in records))
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 5.2))
    figure.suptitle("Scaling with system size", fontsize=13, color=INK, y=0.97)
    figure.text(0.5, 0.915,
                "Restraint count is held to the same per-copy selection, so the "
                "growth is in degrees of freedom, not in data.",
                ha="center", fontsize=8.5, color=INK_SOFT)

    for axis, key, ylabel in ((left, "wall_time", "wall time (s)"),
                              (right, "cpu_time", "CPU time (s)")):
        for sampler in samplers:
            rows = sorted((r for r in records
                           if r["sampler"] == sampler and not r.get("failure")),
                          key=lambda r: r["n_dof"])
            if not rows:
                continue
            colour, marker = style_of(sampler)
            xs = [r["n_dof"] for r in rows]
            ys = [r[key] for r in rows]
            axis.plot(xs, ys, color=colour, marker=marker, markersize=8,
                      linewidth=2.0, markeredgecolor="white", markeredgewidth=1.0)
            axis.annotate(sampler, (xs[-1], ys[-1]), fontsize=7.5, color=INK_SOFT,
                          xytext=(6, 0), textcoords="offset points", va="center",
                          annotation_clip=False)
        axis.set_xlabel("sampled degrees of freedom", color=INK_SOFT, fontsize=9)
        # Log axes only when the data actually spans decades. Over the 2x range
        # a three-copy sweep covers, a log axis buys nothing and its minor tick
        # labels collide; and the right margin has to leave room for the direct
        # labels, which sit outside the last marker.
        _autoscale(axis, [r["n_dof"] for r in records], axis.set_xscale)
        _autoscale(axis, [r[key] for r in records if not r.get("failure")],
                   axis.set_yscale)
        axis.margins(x=0.28, y=0.12)
        _frame(axis, ylabel)

    figure.tight_layout(rect=[0, 0.03, 1, 0.90])
    pdf.savefig(figure)
    plt.close(figure)


def _table_page(pdf, config, records):
    """Every number in the sweep, as text.

    This page is not decoration: three of the series hues fall below 3:1
    contrast on white, and the rule for that is that the data must also be
    reachable without relying on colour.
    """
    figure = plt.figure(figsize=(11, 8.5))
    figure.suptitle("All results", fontsize=13, color=INK, y=0.96)

    header = ["copies", "sampler", "DOF", "restr.", "wall s", "cpu s",
              "RMSD min", "RMSD med", "IMP min", "IMP med", "satisfied", "note"]
    rows = []
    for record in sorted(records, key=lambda r: (r["copy_number"], r["sampler"])):
        def stat(field, fn):
            values = record.get(field)
            if values is None or not len(np.asarray(values)):
                return "-"
            return f"{fn(np.asarray(values, dtype=float)):.4g}"
        rows.append([
            str(record["copy_number"]), record["sampler"], str(record.get("n_dof", "-")),
            str(record.get("n_restraints", "-")),
            f"{record['wall_time']:.1f}",
            "-" if np.isnan(record.get("cpu_time", float("nan"))) else f"{record['cpu_time']:.1f}",
            stat("rmsd", np.min), stat("rmsd", np.median),
            stat("imp_score", np.min), stat("imp_score", np.median),
            "-" if record.get("satisfied") is None
            else f"{100*float(np.max(record['satisfied'])):.0f}%",
            (record.get("failure") or record.get("note") or "")[:22],
        ])

    table = figure.add_subplot(111)
    table.axis("off")
    # Explicit widths: the note column is the only variable-length field, so
    # without them it overflows its cell and runs past the table edge.
    widths = [0.05, 0.11, 0.05, 0.05, 0.06, 0.06,
              0.08, 0.08, 0.09, 0.09, 0.07, 0.18]
    rendered = table.table(cellText=rows, colLabels=header, loc="upper center",
                           cellLoc="center", colWidths=widths)
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(7)
    rendered.scale(1, 1.35)
    for (row, col), cell in rendered.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(color=INK, fontweight="bold")
        else:
            cell.set_text_props(color=INK_SOFT)
            if col == 1:  # a colour chip beside the name, never instead of it
                cell.get_text().set_color(style_of(rows[row - 1][1])[0])
    pdf.savefig(figure)
    plt.close(figure)


def _summary_page(pdf, config, records):
    figure = plt.figure(figsize=(11, 8.5))
    figure.text(0.08, 0.92, config.get("name", "benchmark"), fontsize=22, color=INK)
    figure.text(0.08, 0.875, "IMP + JAX/BlackJAX sampler benchmark",
                fontsize=12, color=INK_SOFT)

    backends = sorted({r.get("backend", "?") for r in records})
    failures = [r for r in records if r.get("failure")]
    lines = [
        f"copy numbers      {config['copy_numbers']}",
        f"samplers          {', '.join(config['samplers'])}",
        f"JAX backend       {', '.join(backends)}",
        f"prior             {config.get('prior')}",
        f"restraints        {config.get('restraints')}",
        f"scoring window    last {config.get('score_window')} frames",
        f"satisfaction tol  {config.get('satisfaction_tolerance')} A",
        f"cases run         {len(records)}"
        + (f"   ({len(failures)} failed)" if failures else ""),
    ]
    for index, line in enumerate(lines):
        figure.text(0.08, 0.79 - 0.035 * index, line, fontsize=10,
                    color=INK_SOFT, family="monospace")

    notes = (
        "Scores on the score pages are computed by IMP on the CPU, by loading each frame back\n"
        "into an IMP model and evaluating the scoring function. They are NOT the BlackJAX\n"
        "log-posterior, which is -S(theta) + log p0(theta) -- a different quantity that would\n"
        "make BlackJAX and IMP samplers incomparable.\n\n"
        "RMSD is superposed, over structured beads only, against the ground-truth structure.\n"
        "The flexible linker is excluded: it has no reference conformation to be right about.\n\n"
        "Every point is one frame from the scoring window. The spread is the result, not noise."
    )
    figure.text(0.08, 0.40, notes, fontsize=9, color=INK_SOFT, va="top")
    if backends == ["cpu"]:
        figure.text(0.08, 0.13,
                    "NOTE: JAX ran on CPU for this sweep, so no CPU-vs-GPU comparison is\n"
                    "possible from these numbers. Re-run on a GPU host to populate it.",
                    fontsize=9, color="#e34948", va="top")
    pdf.savefig(figure)
    plt.close(figure)


def write_report(path: str, config: dict, records: Sequence[dict]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with PdfPages(path) as pdf:
        _summary_page(pdf, config, records)
        _window_panels(
            pdf, records, "rmsd", "RMSD to ground truth (A)",
            "Accuracy: RMSD to the ground-truth structure",
            "Superposed, structured beads only. Lower is better; the short rule is the median.")
        _window_panels(
            pdf, records, "imp_score", "IMP score",
            "IMP score of the final frames",
            "Re-evaluated on the CPU by IMP, not the BlackJAX log-posterior. Lower is better.",
            log=True)
        _window_panels(
            pdf, records, "satisfied", "fraction of restraints satisfied",
            "Restraint satisfaction",
            f"Within {config.get('satisfaction_tolerance')} A of the target distance. "
            "Higher is better.")
        _scaling_page(pdf, records)
        _table_page(pdf, config, records)
    return path
