#!/usr/bin/env python3
"""Matched three-seed comparison of zero-shot, source-only, and A/B/C."""

import argparse
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.summarize_mtda import DOMAINS, mean_std, write_csv

METHODS = ("zero_shot", "source_only", "A", "B", "C")
DELTAS = {"source_only_minus_zero_shot": ("source_only", "zero_shot"),
          "A_minus_zero_shot": ("A", "zero_shot"),
          "B_minus_zero_shot": ("B", "zero_shot"),
          "C_minus_zero_shot": ("C", "zero_shot"),
          "A_minus_source_only": ("A", "source_only"),
          "B_minus_source_only": ("B", "source_only"),
          "C_minus_source_only": ("C", "source_only"),
          "A_minus_C": ("A", "C"), "B_minus_C": ("B", "C"), "A_minus_B": ("A", "B")}


def protocol(cfg):
    return tuple(cfg[key] for key in ("steps", "batch_size", "m1", "m2", "learning_rate",
                                     "weight_decay", "scheduler", "precision", "checkpoint_sha256", "data_root"))


def aggregate(rows, keys):
    result = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key], result[f"{key}_std"] = mean_std(values)
        result[f"{key}_seeds"] = len(values)
    return result


def with_deltas(row):
    for name, (first, second) in DELTAS.items():
        row[name] = row[first] - row[second] if row[first] is not None and row[second] is not None else None
    return row


def summarize_baselines(runs_dir, source_dir, output_dir, seeds=(41, 42, 43)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_paths = sorted(Path(runs_dir).rglob("metrics.json"))
    source_paths = sorted(Path(source_dir).rglob("metrics.json"))
    primary = [json.loads(path.read_text()) for path in primary_paths]
    source_runs = [json.loads(path.read_text()) for path in source_paths]
    primary = [r for r in primary if r["config"]["seed"] in seeds]
    source_runs = [r for r in source_runs if r["config"]["seed"] in seeds]
    configurations = {}
    values, references = {}, {}
    for run in primary:
        cfg = run["config"]
        key = (cfg["seed"], cfg["source"])
        if key in configurations:
            existing = configurations[key]
            if (protocol(cfg) != protocol(existing)
                    or cfg["implementation_sha256"] != existing["implementation_sha256"]
                    or cfg["classnames"] != existing["classnames"]):
                raise ValueError("A/B/C protocol mismatch")
        configurations.setdefault(key, cfg)
        for target, metric in run["targets"].items():
            task = (*key, target)
            if (*task, cfg["variant"]) in values:
                raise ValueError("duplicate A/B/C result")
            values[(*task, cfg["variant"])] = metric["accuracy"] * 100
            if task in references and abs(references[task] - metric["zero_shot"] * 100) > 1e-6:
                raise ValueError("zero-shot references disagree")
            references[task] = metric["zero_shot"] * 100
    for run in source_runs:
        cfg = run["config"]
        key = (cfg["seed"], cfg["source"])
        if cfg["objective"] != "source_cross_entropy_only" or cfg["target_training_images"] != 0:
            raise ValueError("source-only control consumed target training data or another objective")
        if key in configurations:
            baseline = configurations[key]
            if (protocol(cfg) != protocol(baseline) or cfg["classnames"] != baseline["classnames"]
                    or cfg["baseline_implementation_sha256"] != baseline["implementation_sha256"]
                    or cfg["feature_fingerprints"][cfg["source"]] != baseline["feature_fingerprints"][cfg["source"]]):
                raise ValueError("source-only and A/B/C protocols differ")
        for target, metric in run["targets"].items():
            task = (*key, target)
            if key in configurations and target in configurations[key]["feature_fingerprints"]:
                if run["evaluation_fingerprints"][target] != configurations[key]["feature_fingerprints"][target]:
                    raise ValueError("source-only target evaluation uses different image features")
            if (*task, "source_only") in values:
                raise ValueError("duplicate source-only result")
            if task in references and abs(references[task] - metric["zero_shot"] * 100) > 1e-6:
                raise ValueError("source-only evaluation uses a different zero-shot reference")
            references[task] = metric["zero_shot"] * 100
            values[(*task, "source_only")] = metric["accuracy"] * 100
    detail = []
    for seed in seeds:
        for source in DOMAINS:
            for target in DOMAINS:
                if source == target:
                    continue
                task = (seed, source, target)
                row = {"seed": seed, "source": source, "target": target,
                       "zero_shot": references.get(task)}
                row.update({method: values.get((*task, method)) for method in METHODS[1:]})
                detail.append(with_deltas(row))
    tasks = []
    for source in DOMAINS:
        for target in DOMAINS:
            if source != target:
                selected = [r for r in detail if r["source"] == source and r["target"] == target]
                tasks.append({"source": source, "target": target, **aggregate(selected, (*METHODS, *DELTAS))})
    macros, by_source, by_target, paired = [], [], [], []
    for axis, destination in (("source", by_source), ("target", by_target)):
        for domain in (*DOMAINS, "All"):
            seed_rows = []
            for seed in seeds:
                selected = [r for r in detail if r["seed"] == seed and (domain == "All" or r[axis] == domain)]
                row = {"seed": seed, axis: domain}
                for method in METHODS:
                    scores = [r[method] for r in selected]
                    row[method] = statistics.mean(scores) if all(v is not None for v in scores) else None
                with_deltas(row)
                seed_rows.append(row)
                macros.append({"group": axis, "domain": domain, "seed": seed,
                               **{key: row[key] for key in (*METHODS, *DELTAS)}})
            destination.append({axis: domain, **aggregate(seed_rows, (*METHODS, *DELTAS))})
            paired.append({"group": axis, "domain": domain, **aggregate(seed_rows, tuple(DELTAS))})
    write_csv(output_dir / "baseline_per_seed.csv", detail)
    write_csv(output_dir / "baseline_tasks.csv", tasks)
    write_csv(output_dir / "baseline_macro_per_seed.csv", macros)
    write_csv(output_dir / "baseline_source_summary.csv", by_source)
    write_csv(output_dir / "baseline_target_summary.csv", by_target)
    write_csv(output_dir / "baseline_paired_differences.csv", paired)
    resources = []
    for method in ("A", "B", "C", "source_only"):
        selected = source_runs if method == "source_only" else [r for r in primary if r["config"]["variant"] == method]
        resources.append({"method": method, "completed_runs": len(selected),
                          "training_seconds": sum(r["training_seconds"] for r in selected),
                          "peak_memory_bytes": max((r["peak_memory_bytes"] for r in selected), default=0)})
    write_csv(output_dir / "baseline_resources.csv", resources)
    overall = by_source[-1]
    complete = all(overall[f"{method}_seeds"] == len(seeds) for method in METHODS)
    def display(row, method):
        value, deviation = row[method], row[f"{method}_std"]
        if value is None:
            return "pending"
        text = f"{value:.2f}" + (f" ± {deviation:.2f}" if deviation is not None else "")
        if row[f"{method}_seeds"] != len(seeds):
            text += f" ({row[f'{method}_seeds']}/{len(seeds)} seeds)"
        return text
    lines = ["# Office-Home multi-seed baseline consolidation", "",
             f"Requested seeds: **{list(seeds)}**. A/B/C completed runs: **{len(primary)}/{20 * len(seeds)}**. "
             f"Source Prompt Only completed runs: **{len(source_runs)}/{4 * len(seeds)}**.", "",
             "All methods use the same frozen RN50, class order, image feature caches and full-batch-tail "
             "evaluation. Transductive A/B/C train on unlabeled target images; Source Prompt Only loads "
             "target images exclusively after source-supervised training, for evaluation. Main accuracy "
             "uses the fixed update-1000 checkpoint, with no target-accuracy selection.", "",
             "Source Prompt Only independently learns the class context and source-domain context with "
             "Source CE only. It evaluates **the source text prototypes** on every target. Initialization, "
             "source permutations, batch size, AdamW/0.003, weight decay, and per-update cosine match A/B/C. "
             "Unused target domain contexts are frozen and never used as predictors. It is not the "
             "target-influenced source prompt from A/C.", "",
             "Zero-shot uses `A photo of a [CLASS]` and no trained prompts. Its accuracy is deterministic "
             "for this fixed dataset/checkpoint; repeating it across seeds does not estimate dataset "
             "uncertainty.", "",
             "## Source and overall macro accuracy (%)", "",
             "| Source | CLIP Zero-shot | Source Prompt Only | A | B | C |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in by_source:
        lines.append("| " + " | ".join([row["source"], *[display(row, method) for method in METHODS]]) + " |")
    lines += ["", "## Accuracy grouped by target (%)", "",
              "Each target mean averages its three different sources per seed, then averages seeds.", "",
              "| Target | CLIP Zero-shot | Source Prompt Only | A | B | C |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in by_target:
        lines.append("| " + " | ".join([row["target"], *[display(row, method) for method in METHODS]]) + " |")
    lines += ["", "## All 12 directed tasks (%)", "",
              "| Source | Target | CLIP Zero-shot | Source Prompt Only | A | B | C |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in tasks:
        lines.append("| " + " | ".join([row["source"], row["target"], *[display(row, method) for method in METHODS]]) + " |")
    lines += ["", "Means and sample standard deviations are computed **across seed-level macro scores**, "
              "not across images or by pooling tasks. Paired deltas are computed within each seed first. "
              "Missing seed/task results remain pending; aggregate differences are not final until the "
              "requested matrix and source-only controls are complete.", "", "## Interpretation and attribution", ""]
    if complete:
        for name in DELTAS:
            lines.append(f"- {name}: {display(overall, name)} percentage points.")
        lines += ["", "Source Only − Zero-shot measures the effect of source supervision together with "
                  "learning the source prompt. A/B/C − Source Only measures the additional target-adaptation "
                  "pipeline, which changes both the target-training signal and inference prompt. It does "
                  "not isolate a universal pure-SPL effect independently of prompt design. B is the cleanest "
                  "source-teacher control: its source contexts receive Source CE only, while its target "
                  "contexts receive only SPL. A/B/C comparisons test their prescribed sharing choices.", "",
                  "Three seeds quantify initialization/sampling variability under this protocol; they "
                  "do not establish statistical significance, generalization beyond Office-Home, or "
                  "a causal benefit from the single-source bank (its SPL weights are identically one)."]
    else:
        lines.append("The requested consolidation is incomplete; full comparisons remain pending.")
    lines += ["", "## Provenance and resources", "",
              "| Method | Completed runs | Sum of training seconds | Peak allocated GiB |",
              "|---|---:|---:|---:|"]
    for row in resources:
        lines.append(f"| {row['method']} | {row['completed_runs']} | {row['training_seconds']:.1f} | "
                     f"{row['peak_memory_bytes'] / 2**30:.2f} |")
    if primary:
        cfg = primary[0]["config"]
        lines += ["", f"Upstream commit: `{cfg['upstream_sha']}`. Unchanged A/B/C training implementation "
                  f"SHA256: `{cfg['implementation_sha256']}`. Each run preserves configuration, seed, "
                  "checkpoint, history, and its actual command in `environment.json`.", "",
                  f"Data root: `{cfg['data_root']}`; CLIP checkpoint SHA256: `{cfg['checkpoint_sha256']}`."]
    lines += ["", "The original seed-42 report is preserved under `reports/seed_42/`. "
              "Literature research is stored in `docs/spl_reliability_literature.md`; no geometry-based "
              "SPL method, target graph, new bank weighting, or exploratory experiment was implemented.", ""]
    (output_dir / "baseline_report.md").write_text("\n".join(lines))
    print(f"Baseline consolidation: ABC {len(primary)}/{20 * len(seeds)}, SourceOnly {len(source_runs)}/{4 * len(seeds)}, complete={complete}")
    return {"complete": complete, "abc_runs": len(primary), "source_runs": len(source_runs),
            "detail": detail, "tasks": tasks, "source_summary": by_source, "target_summary": by_target}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/mtda_spl")
    cli.add_argument("--source-runs-dir", default="runs/source_prompt_only")
    cli.add_argument("--output-dir", default="reports/multiseed")
    cli.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    args = cli.parse_args()
    summarize_baselines(args.runs_dir, args.source_runs_dir, args.output_dir, tuple(args.seeds))


if __name__ == "__main__":
    main()
