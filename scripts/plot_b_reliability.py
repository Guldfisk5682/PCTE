#!/usr/bin/env python3
"""Export final accuracy and Clipart update-aligned curves as standalone plots."""

import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report-dir", default="reports/b_reliability")
    args = cli.parse_args()
    root = Path(args.report_dir)
    methods = ("R0", "R0M", "R1", "R2", "R3", "R4", "R5", "R6")
    rows = list(csv.DictReader((root / "source_summary.csv").open()))
    all_row = next(r for r in rows if r["source"] == "All")
    present = [m for m in methods if all_row.get(m)]
    colors = plt.get_cmap("tab10")(np.arange(len(methods)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    scores = [float(all_row[m]) for m in present]
    errors = [float(all_row.get(m + "_std") or 0) for m in present]
    ax.bar(present, scores, yerr=errors, capsize=3, color=[colors[methods.index(m)] for m in present])
    ax.set_ylabel("Macro target accuracy (%)")
    ax.set_title("Fixed final-budget accuracy")
    ax.set_ylim(max(0, min(scores) - 2), max(scores) + 1)
    for i, value in enumerate(scores):
        ax.text(i, value + errors[i] + .06, f"{value:.2f}", ha="center", fontsize=8)
    points = list(csv.DictReader((root / "accuracy_curves.csv").open()))
    grouped = {}
    import json
    for point in points:
        if point["target"] != "Clipart" or point["variant"] == "R0":
            continue
        updates = json.loads(point["effective_updates"])["Clipart"]
        grouped.setdefault((point["variant"], point["source"], point["seed"]), {})[updates] = float(point["accuracy"])
    ax = axes[1]
    for method in methods[1:]:
        series = [values for (m, _, _), values in grouped.items() if m == method]
        if not series:
            continue
        end = min(max(values) for values in series)
        grid = np.linspace(0, end, 101)
        interpolated = [np.interp(grid, sorted(values), [values[x] for x in sorted(values)]) for values in series]
        ax.plot(grid, np.mean(interpolated, axis=0), color=colors[methods.index(method)], label=method)
    ax.set_xlabel("Effective Clipart student optimizer updates")
    ax.set_ylabel("Clipart accuracy (%)")
    ax.set_title("Mean over sources/seeds; linear interpolation")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "accuracy_and_clipart_curves.png", dpi=180)
    fig.savefig(root / "accuracy_and_clipart_curves.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
