#!/usr/bin/env python3
"""Summarize all directed tasks, preserving missing results and seed variance."""

import argparse
import csv
import json
from pathlib import Path
import statistics

DOMAINS = ("Art", "Clipart", "Product", "Real World")


def mean_std(values):
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None) if values else (None, None)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs_dir, output_dir):
    runs_dir, output_dir = Path(runs_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(path.read_text()) for path in sorted(runs_dir.rglob("metrics.json"))]
    configured = [json.loads(path.read_text()) for path in sorted(runs_dir.rglob("config.json"))]
    seeds = sorted({run["config"]["seed"] for run in runs} | {cfg["seed"] for cfg in configured})
    scores, references, commands = {}, {}, []
    protocols = set()
    for path in sorted(runs_dir.rglob("environment.json")):
        environment = json.loads(path.read_text())
        commands.extend(environment.get("invocations", [environment["command"]]))
    for run in runs:
        cfg = run["config"]
        protocols.add((cfg["steps"], cfg["batch_size"], cfg["m1"], cfg["m2"], cfg["learning_rate"],
                       cfg["lambda_target"], cfg["checkpoint_sha256"], cfg["implementation_sha256"]))
        for target, metric in run["targets"].items():
            key = (cfg["seed"], cfg["source"], target, cfg["variant"])
            if key in scores:
                raise ValueError(f"duplicate task result: {key}")
            scores[key] = 100 * metric["accuracy"]
            reference_key = key[:3]
            if reference_key in references and abs(references[reference_key] - 100 * metric["zero_shot"]) > 1e-6:
                raise ValueError("zero-shot reference differs across variants")
            references[reference_key] = 100 * metric["zero_shot"]
    if len(protocols) > 1:
        raise ValueError("incompatible training configurations or code versions in one summary")
    detail = []
    for seed in seeds:
        for source in DOMAINS:
            for target in DOMAINS:
                if source == target:
                    continue
                row = {"seed": seed, "source": source, "target": target,
                       "zero_shot": references.get((seed, source, target))}
                for variant in "ABC":
                    row[variant] = scores.get((seed, source, target, variant))
                row["A_minus_C"] = row["A"] - row["C"] if row["A"] is not None and row["C"] is not None else None
                row["B_minus_C"] = row["B"] - row["C"] if row["B"] is not None and row["C"] is not None else None
                detail.append(row)
    aggregated = []
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            subset = [row for row in detail if row["source"] == source and row["target"] == target]
            row = {"source": source, "target": target}
            for key in ("zero_shot", "A", "B", "C", "A_minus_C", "B_minus_C"):
                row[key], row[f"{key}_std"] = mean_std([r[key] for r in subset if r[key] is not None])
                row[f"{key}_seeds"] = sum(r[key] is not None for r in subset)
            aggregated.append(row)
    summary = []
    for source in (*DOMAINS, "All"):
        row = {"source": source}
        for variant in ("zero_shot", "A", "B", "C"):
            seed_means = []
            for seed in seeds:
                selected = [r for r in detail if r["seed"] == seed and (source == "All" or r["source"] == source)]
                if selected and all(r[variant] is not None for r in selected):
                    seed_means.append(statistics.mean(r[variant] for r in selected))
            row[variant], row[f"{variant}_std"] = mean_std(seed_means)
            row[f"{variant}_complete_seeds"] = len(seed_means)
        summary.append(row)
    resources = []
    for variant in "ABC":
        selected = [r for r in runs if r["config"]["variant"] == variant]
        resources.append({"variant": variant, "completed_runs": len(selected),
                          "training_seconds": sum(r["training_seconds"] for r in selected),
                          "peak_memory_bytes": max((r["peak_memory_bytes"] for r in selected), default=0)})
    write_csv(output_dir / "officehome_mtda_per_seed.csv", detail)
    write_csv(output_dir / "officehome_mtda_A_B_C.csv", aggregated)
    write_csv(output_dir / "officehome_mtda_summary.csv", summary)
    write_csv(output_dir / "officehome_mtda_resources.csv", resources)
    expected = len(seeds) * 20
    lines = ["# Office-Home CRPL → MTDA SPL-only Report", "",
             f"Completed training runs: **{len(runs)}/{expected}**, seeds: {seeds or 'none'}. "
             f"Directed task results: {len(scores)}/{len(seeds) * 36}.", "",
             "Protocol: transductive UDA; target training sees images without class labels. "
             "Accuracy uses every target image and its own current target prompt from the fixed final checkpoint. "
             "No target-accuracy checkpoint selection is performed.", ""]
    if runs:
        cfg = runs[0]["config"]
        lines += [f"Upstream: https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning, "
                  f"commit `{cfg['upstream_sha']}`. Official remote cloning encountered TLS failures; "
                  "the unmodified local upstream Git bundle was cloned instead (see implementation notes).", "",
                  f"Configuration: {cfg['steps']} updates, batch size {cfg['batch_size']} per domain, "
                  f"M1/M2={cfg['m1']}/{cfg['m2']}, float32 CLIP, AdamW LR {cfg['learning_rate']}, "
                  f"weight decay {cfg['weight_decay']}, lambda_T={cfg['lambda_target']}, cosine once per update.", "",
                  f"Data: `{cfg['data_root']}`. Weights: `{cfg['clip_checkpoint']}` "
                  f"(SHA256 `{cfg['checkpoint_sha256']}`). Class order is defined by source directories.", ""]
        environments = [json.loads(p.read_text()) for p in sorted(runs_dir.rglob("environment.json"))]
        if environments:
            env = environments[0]
            lines += [f"Environment: Python {env['python'].split()[0]}, PyTorch {env['torch']}, "
                      f"CUDA {env['cuda']}, GPU {env['gpu']}. Runtime details are saved per run.", ""]
    lines += ["## Results (%)", "", "| Source | Target | Zero-shot | A | B | C | A−C | B−C |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    def display(value):
        return f"{value:.2f}" if value is not None else "pending"
    for row in aggregated:
        lines.append("| " + " | ".join([row["source"], row["target"], *[
            display(row[key]) for key in ("zero_shot", "A", "B", "C", "A_minus_C", "B_minus_C")]]) + " |")
    lines += ["", "## Source and overall macro means", "", "| Source | Zero-shot | A | B | C |",
              "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| " + " | ".join([row["source"], *[display(row[v]) + (
            f" ± {row[v + '_std']:.2f}" if row[v + "_complete_seeds"] > 1 else "")
            for v in ("zero_shot", "A", "B", "C")]]) + " |")
    lines += ["", "A source mean requires all three targets; the overall macro requires all 12 directed tasks "
              "for that seed. CSV files include sample standard deviations across seeds and completeness counts.", "",
              "## Implementation and resources", "",
              "A shares a single class context across all four domains. B has four independent class contexts; "
              "its source teacher is detached from each target loss. C resets prompts, optimizer, scheduler, "
              "bank, batch streams, and RNG for every pair, sharing class context within the pair. "
              "All domains have independent domain contexts with reproducible initialization.", "",
              "SPL fuses base and the sole learned source similarities with weights 0.5/0.5 before applying "
              "the frozen CLIP scale and softmax. The multi-source combined branch is omitted. "
              "Only source CE and mean target soft CE are optimized; no OT, hard pseudo-label CE, "
              "entropy, filtering, or cross-target objective is used.", "",
              "Frozen image outputs are cached using identical deterministic CLIP preprocessing; "
              "text features are recomputed from current trainable prompts. Cosine advances once per update, "
              "correcting upstream's inconsistent cadence. AdamW/0.003 follows official source defaults "
              "rather than the paper's SGD/0.005 description.", "",
              "| Variant | Completed runs | Training seconds (sum) | Peak allocated GPU GiB |",
              "|---|---:|---:|---:|"]
    for row in resources:
        lines.append(f"| {row['variant']} | {row['completed_runs']} | {row['training_seconds']:.1f} | "
                     f"{row['peak_memory_bytes'] / 2**30:.2f} |")
    lines += ["", "Training timing excludes feature extraction and final evaluation; summed device time differs "
              "from matrix wall time when runs use both GPUs. Every target receives the same number of updates. "
              "C trains the source teacher separately three times per source.", "",
              "## Source bank and interpretation", "",
              "The bank is a cumulative mean of unnormalized source features, saved with integer counts. "
              "Single-source softmax weights equal 1 for every class and sample, so the bank is statistical "
              "state and does not change these SPL targets. Coverage and sample counts are recorded each step.", ""]
    for variant in "ABC":
        banks = [run["bank"] for run in runs if run["config"]["variant"] == variant]
        if banks:
            lines.append(f"{variant}: observed classes {min(b['observed_classes'] for b in banks)}–"
                         f"{max(b['observed_classes'] for b in banks)}/{banks[0]['classes']}; "
                         f"source samples per run {min(b['samples'] for b in banks)}–{max(b['samples'] for b in banks)}.")
    overall = summary[-1]
    if seeds and all(overall[v] is not None and overall[v + "_complete_seeds"] == len(seeds) for v in "ABC"):
        lines += ["", f"Observed A−C macro difference: {overall['A'] - overall['C']:.2f} percentage points; "
                  f"B−C: {overall['B'] - overall['C']:.2f}; A−B: {overall['A'] - overall['B']:.2f}. "
                  f"Against the zero-shot macro ({overall['zero_shot']:.2f}%), "
                  f"A/B/C change by {overall['A'] - overall['zero_shot']:+.2f}/"
                  f"{overall['B'] - overall['zero_shot']:+.2f}/"
                  f"{overall['C'] - overall['zero_shot']:+.2f} percentage points. "
                  "These measured differences do not isolate "
                  "source-teacher training count from class-context sharing. Single-seed results provide "
                  "no estimate of variability across seeds."]
        best_variant = max("ABC", key=lambda v: overall[v])
        lines += ["", f"{best_variant} has the highest observed macro in this experiment. "
                  f"B and C differ by {abs(overall['B'] - overall['C']):.2f} percentage points in macro, "
                  "although individual tasks can favor different variants. The direct SPL migration "
                  "is operational; the measured gains and task differences are listed above."]
        if len(seeds) == 1:
            lines += ["", "This one-seed result does not establish a robust advantage from sharing "
                      "class context across targets. Additional seeds are needed to assess the "
                      "stability of the small differences between A/B/C."]
        timings = {row["variant"]: row["training_seconds"] for row in resources}
        if timings["A"] and timings["B"]:
            lines += ["", f"C consumes {timings['C'] / timings['A']:.2f}× A's and "
                      f"{timings['C'] / timings['B']:.2f}× B's summed training time. "
                      "This includes its repeated source-prompt optimization across independent pairs."]
    else:
        lines += ["", "The matrix is incomplete; the effectiveness of MTDA and A/B/C ranking are not yet established."]
    missing = [row for row in detail if any(row[v] is None for v in "ABC")]
    lines += ["", "## Execution and remaining work", "",
              f"Pending task/seed rows: {len(missing)}. Inspect `matrix.log`, per-job `.log` files, "
              "and each run's `train.log` for failures. Re-run the matrix with the same arguments and "
              "`--retry-failed` to skip successful runs and restore checkpoints for interrupted runs.", "",
              "Actually executed commands recorded in run environments:", "", "```text"]
    import shlex
    lines += [shlex.join(command) for command in commands]
    lines += ["```", "", "## Future Work", "",
              "After the fixed baselines finish, consider additional seeds and separately designed bank or "
              "cross-target ablations. No such algorithms are implemented in this experiment.", ""]
    (output_dir / "officehome_mtda_report.md").write_text("\n".join(lines))
    print(f"Summarized {len(runs)}/{expected} runs and {len(scores)} directed task results into {output_dir}")
    return {"runs": len(runs), "task_results": len(scores), "tasks": aggregated, "summary": summary}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/mtda_spl")
    cli.add_argument("--output-dir", default="reports")
    args = cli.parse_args()
    summarize(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
