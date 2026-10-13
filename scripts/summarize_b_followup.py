#!/usr/bin/env python3
"""Complete four-source follow-up reports, paired seeds, and fixed-teacher diagnostics."""

import argparse
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from mtda.experiment import write_json
from mtda.reliability_followup.references import METHODS
from scripts.summarize_b_reliability import flat_csv

DOMAINS = ("Art", "Clipart", "Product", "Real World")


def average(values):
    return statistics.mean(values) if values else None


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize(runs, baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    tasks, resources, diagnostics, actual = [], [], [], []
    for root_dir, methods in ((baseline, ("R0M", "R2", "R4", "R5")), (runs, tuple(METHODS))):
        for method in methods:
            for path in sorted((root_dir / method).rglob("metrics.json")):
                record = json.loads(path.read_text())
                cfg = record["config"]
                if record["effective_updates"] != {d: cfg["steps"] for d in [cfg["source"], *cfg["targets"]]}:
                    raise ValueError("unequal effective update budget")
                if cfg["target_training_ground_truth"] or cfg["checkpoint_selection"] != "fixed final budget":
                    raise ValueError("target-label feedback forbidden")
                for target, result in record["targets"].items():
                    tasks.append({"method": method, "source": cfg["source"], "target": target,
                                  "seed": cfg["seed"], "accuracy": 100 * result["accuracy"]})
                if root_dir == runs:
                    resources.append({"method": method, "source": cfg["source"], "seed": cfg["seed"],
                                      "student_updates": record["effective_updates"],
                                      "teacher_preparation_updates_per_branch": cfg["teacher_preparation_budget_per_branch"],
                                      "optimizer_step_calls": record["optimizer_step_calls"],
                                      "sample_exposures": record["sample_exposures"],
                                      "training_seconds": record["training_seconds"],
                                      "reference_lookahead_control": cfg["reference_provenance"].get("lookahead_control", False)})
                    diagnostics.extend(read_lines(path.parent / "diagnostics.jsonl"))
                    p = path.parent / "actual_teacher_contributions.json"
                    if p.exists():
                        actual.extend(json.loads(p.read_text()))
    values = {(r["method"], r["source"], r["target"], r["seed"]): r["accuracy"] for r in tasks}
    if len(values) != len(tasks):
        raise ValueError("duplicate task result")
    macro, sources = [], []
    for method in dict.fromkeys(r["method"] for r in tasks):
        for seed in sorted({r["seed"] for r in tasks if r["method"] == method}):
            keys = [(method, s, t, seed) for s in DOMAINS for t in DOMAINS if s != t]
            if not all(k in values for k in keys):
                continue
            macro.append({"method": method, "seed": seed, "macro_accuracy": average([values[k] for k in keys])})
            for s in DOMAINS:
                sources.append({"method": method, "source": s, "seed": seed,
                                "macro_accuracy": average([values[method, s, t, seed] for t in DOMAINS if t != s])})
    seed_values = {(r["method"], r["seed"]): r["macro_accuracy"] for r in macro}
    paired = []
    for method in ("R2", "R4", *METHODS):
        control = "R0M" if method in ("R2", "R4") else "R4"
        seeds = sorted(seed for m, seed in seed_values if m == method and (control, seed) in seed_values)
        differences = [seed_values[method, seed] - seed_values[control, seed] for seed in seeds]
        if differences:
            paired.append({"method": method, "control": control, "seeds": seeds,
                           "mean_gain_pp": average(differences),
                           "std_gain_pp": statistics.stdev(differences) if len(differences) > 1 else None,
                           "per_seed_gain_pp": dict(zip(seeds, differences))})
    bank_rows = []
    for method in METHODS:
        rows = [r for r in diagnostics if r["variant"] == method and r["event"] == "final"]
        if not rows:
            continue
        for target_group in ("all", "Clipart"):
            selected = [r for r in rows if target_group == "all" or r["target"] == target_group]
            active = [r for r in actual if r["variant"] == method and r["teacher_count"] > 1
                      and (target_group == "all" or r["target"] == target_group)]
            bank_rows.append({"method": method, "target_group": target_group,
                              "final_mean_weight_entropy": average([r["contribution"]["weight_entropy_multi_valid"] for r in selected
                                                                     if r["contribution"]["weight_entropy_multi_valid"] is not None]),
                              "final_one_hot_rate": average([r["contribution"]["one_hot_multi_valid_class_sample_fraction"] for r in selected
                                                             if r["contribution"]["one_hot_multi_valid_class_sample_fraction"] is not None]),
                              "final_source_weight": average([r["contribution"]["mean_weights"]["Source"] for r in selected]),
                              "active_training_source_weight": average([r["mean_weights"]["Source"] for r in active]),
                              "active_training_one_hot_rate": average([r["one_hot_multi_valid_class_sample_fraction"] for r in active
                                                                      if r["one_hot_multi_valid_class_sample_fraction"] is not None]),
                              "active_training_records": len(active)})
    candidates = []
    for method in METHODS:
        if method == "FUniform" or (method, 42) not in seed_values:
            continue
        delta = seed_values[method, 42] - seed_values["R4", 42]
        positive = sum(next(r["macro_accuracy"] for r in sources if r["method"] == method and r["seed"] == 42 and r["source"] == s)
                       > next(r["macro_accuracy"] for r in sources if r["method"] == "R4" and r["seed"] == 42 and r["source"] == s)
                       for s in DOMAINS)
        candidates.append({"method": method, "gain_pp": delta, "positive_sources": positive,
                           "qualifies_multiseed": delta >= .5 and positive >= 3})
    for name, rows in (("accuracy_tasks", tasks), ("macro_per_seed", macro), ("source_summary", sources),
                       ("paired_comparisons", paired), ("resources", resources), ("bank_weight_diagnostics", bank_rows),
                       ("actual_teacher_contributions", actual)):
        flat_csv(output / f"{name}.csv", rows)
    write_json(output / "summary.json", {"macro": macro, "paired": paired, "bank_weights": bank_rows,
                                           "candidates": candidates, "completed_new_runs": len(resources)})
    return macro, paired, bank_rows, candidates


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/b_followup")
    cli.add_argument("--baseline-dir", default="runs/b_reliability")
    cli.add_argument("--output-dir", default="reports/b_followup")
    args = cli.parse_args()
    macro, _, _, candidates = summarize(Path(args.runs_dir), Path(args.baseline_dir), Path(args.output_dir))
    print(json.dumps({"complete_macro_records": len(macro), "candidates": candidates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
