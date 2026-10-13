#!/usr/bin/env python3
"""Publication-style temperature and admission figures from completed results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reports-dir", default="reports/b_followup")
    args = cli.parse_args()
    root = Path(args.reports_dir)
    report = json.loads((root / "summary.json").read_text())
    values = {(r["method"],r["seed"]):r["macro_accuracy"] for r in report["macro"]}
    methods = ("FBeta0","FBeta0p1","FBeta1","FBeta10")
    if not all((m,42) in values for m in methods):
        raise ValueError("temperature scan is incomplete")
    weights = {r["method"]:r for r in report["bank_weights"] if r["target_group"]=="all"}
    clip = {r["method"]:r for r in report["bank_weights"] if r["target_group"]=="Clipart"}
    fig,axes = plt.subplots(1,3,figsize=(13.5,3.9),layout="constrained")
    x = range(4)
    axes[0].plot(x,[values[m,42] for m in methods],"o-",label="Fixed R4 teachers + bank")
    axes[0].axhline(values["R4",42],color="black",linestyle="--",label="R4 uniform")
    axes[0].axhline(values["R5",42],color="gray",linestyle=":",label="Original R5, beta=10")
    axes[0].set_ylabel("12-task macro accuracy (%)")
    axes[0].set_title("Distance temperature, seed 42")
    axes[0].legend(fontsize=8)
    axes[1].plot(x,[100*weights[m]["final_one_hot_rate"] for m in methods],"o-",label="Near one-hot (valid multi-domain)")
    axes[1].plot(x,[100*clip[m]["final_source_weight"] for m in methods],"s-",label="Source weight, Clipart target")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0,100)
    axes[1].set_title("Concentration and Source preference")
    axes[1].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(list(x),["0","0.1","1","10"])
        ax.set_xlabel("Beta (shared teacher snapshots and banks)")
    admitted = ("R4","FStartInit","FStartRef")
    labels = ("Progressive\nR4 online","All at start\ninitial teachers","All at start\ntrained references*")
    bars = axes[2].bar(range(3),[values[m,42] for m in admitted],color=["#4c78a8","#f28e2b","#59a14f"])
    axes[2].set_xticks(range(3),labels,fontsize=8)
    axes[2].set_ylabel("12-task macro accuracy (%)")
    axes[2].set_ylim(min(values[m,42] for m in admitted)-.8,max(values[m,42] for m in admitted)+.6)
    axes[2].bar_label(bars,fmt="%.3f",padding=3,fontsize=9)
    axes[2].set_title("Admission timing and teacher maturity")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Office-Home B follow-up; equal student update budgets\n*Offline two-stage timing control; trained reference preparation adds shared teacher compute",fontsize=11)
    for suffix in ("png","pdf"):
        fig.savefig(root/f"temperature_admission.{suffix}",dpi=220,bbox_inches="tight")


if __name__=="__main__":
    main()
