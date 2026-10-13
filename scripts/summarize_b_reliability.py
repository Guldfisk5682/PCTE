#!/usr/bin/env python3
"""Complete-task accuracy, budget, stage, pseudo-label and bank diagnostics."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.summarize_mtda import DOMAINS, mean_std, write_csv

METHODS = ("R0", "R0M", "R1", "R2", "R3", "R4", "R5", "R6")
COMPARISONS = {"E2H": ("R1", "R0M"), "E2H_uniform": ("R2", "R1"),
               "E2H_bank": ("R3", "R2"), "joint_uniform": ("R4", "R0M"),
               "joint_bank": ("R5", "R4"), "refresh": ("R6", "R5"), "matched_control": ("R0M", "R0")}


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def flat_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(path, [{k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (list, dict))
                    else row.get(k) for k in keys} for row in rows])


def summarize(runs_dir, r0_dir, output_dir, seeds=(42,), sources=DOMAINS, selected_methods=METHODS):
    methods = tuple(selected_methods)
    if not methods or not set(methods) <= set(METHODS):
        raise ValueError("unknown or empty report method selection")
    comparisons_spec = {q: p for q, p in COMPARISONS.items() if set(p) <= set(methods)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in sorted(Path(runs_dir).rglob("metrics.json")):
        run = json.loads(path.read_text())
        cfg = run["config"]
        if cfg["variant"] in methods and cfg["seed"] in seeds and cfg["source"] in sources:
            runs.append((cfg["variant"], path.parent, run))
    for source in sources:
        for seed in seeds:
            root = Path(r0_dir) / "B" / source.replace(" ", "_") / f"seed_{seed}"
            if "R0" in methods and (root / "metrics.json").exists():
                runs.append(("R0", root, json.loads((root / "metrics.json").read_text())))
    values, resources, orders = {}, [], {}
    curves, evaluations, diagnostics, events, offline_points, offline_banks, offline_teachers = [], [], [], [], [], [], []
    actual_contributions = []
    configs, protocols = {}, set()
    for method, root, run in runs:
        cfg, source, seed = run["config"], run["config"]["source"], run["config"]["seed"]
        protocol = tuple(cfg[k] for k in ("steps", "batch_size", "m1", "m2", "learning_rate", "lambda_target", "checkpoint_sha256", "data_root"))
        protocols.add(protocol)
        configs[(method, source, seed)] = cfg
        for target, result in run["targets"].items():
            key = (method, source, target, seed)
            if key in values:
                raise ValueError(f"duplicate reliability result: {key}")
            values[key] = result["accuracy"] * 100
        if method != "R0":
            if run["effective_updates"] != {d: cfg["steps"] for d in [source, *cfg["targets"]]}:
                raise ValueError("effective update budget mismatch")
            if cfg["target_training_ground_truth"] or cfg["checkpoint_selection"] != "fixed final budget":
                raise ValueError("training ground-truth or checkpoint-selection contract violated")
            order = cfg["domain_order"]
            if source in orders and orders[source] != order:
                raise ValueError("domain order changed across variants or seeds")
            orders[source] = order
        resources.append({"variant": method, "source": source, "seed": seed,
                          "global_steps": run.get("global_steps", cfg["steps"]),
                          "source_updates": run.get("effective_updates", {}).get(source, cfg["steps"]),
                          "target_updates": {d: run.get("effective_updates", {}).get(d, cfg["steps"]) for d in run["targets"]},
                          "optimizer_step_calls": run.get("optimizer_step_calls", cfg["steps"]),
                          "target_sample_exposures": {d: run.get("sample_exposures", {}).get(d, cfg["steps"] * cfg["batch_size"]) for d in run["targets"]},
                          "source_sample_exposures": run.get("sample_exposures", {}).get(source, cfg["steps"] * cfg["batch_size"]),
                          "elapsed_seconds": run["training_seconds"], "peak_memory_bytes": run["peak_memory_bytes"]})
        evaluations.extend(read_events(root / "evaluations.jsonl"))
        diagnostics.extend(read_events(root / "diagnostics.jsonl"))
        events.extend(read_events(root / "events.jsonl"))
        actual_path = root / "actual_teacher_contributions.json"
        if actual_path.exists():
            actual_contributions.extend(json.loads(actual_path.read_text()))
        for name, destination in (("pseudo_label_quality", offline_points), ("bank_quality", offline_banks),
                                   ("admitted_teacher_quality", offline_teachers)):
            path = root / "offline_analysis" / f"{name}.json"
            if path.exists():
                destination.extend(json.loads(path.read_text()))
        # Downsample loss curves by fixed 100-global-step windows; raw updates remain in each run.
        path = root / "history.csv"
        if path.exists():
            with path.open() as handle:
                history = list(csv.DictReader(handle))
            windows = {}
            for row in history:
                rows = [row] if method != "R0" else [
                    {**row, "target": d, "target_spl": loss, "source_ce": row["source_loss"],
                     "total_loss": row["loss"], "stage": 0, "source_update_count": row["step"],
                     "target_update_count": row["step"]} for d, loss in json.loads(row["target_losses"]).items()]
                for item in rows:
                    windows.setdefault((item["target"], (int(item["step"]) - 1) // 100), []).append(item)
            for (target, _), selected in windows.items():
                last = selected[-1]
                source_losses = [float(r["source_ce"]) for r in selected if r["source_ce"] not in (None, "")]
                curves.append({"variant": method, "source": source, "target": target, "seed": seed,
                               "step": int(last["step"]), "stage": int(last["stage"]),
                               "source_ce": statistics.mean(source_losses) if source_losses else None,
                               "target_spl": statistics.mean(float(r["target_spl"]) for r in selected),
                               "total_loss": statistics.mean(float(r["total_loss"]) for r in selected),
                               "lr": float(last["lr"]), "source_updates": int(last["source_update_count"]),
                               "target_updates": int(last["target_update_count"])})
    if len(protocols) > 1:
        raise ValueError("mixed backbone/budget/optimizer hyperparameters in report")
    detail, tasks = [], []
    for source in sources:
        for target in DOMAINS:
            if target == source:
                continue
            rows = []
            for seed in seeds:
                row = {"source": source, "target": target, "seed": seed,
                       **{m: values.get((m, source, target, seed)) for m in methods}}
                detail.append(row)
                rows.append(row)
            task = {"source": source, "target": target}
            for method in methods:
                valid = [r[method] for r in rows if r[method] is not None]
                task[method], task[method + "_std"] = mean_std(valid)
                task[method + "_seeds"] = len(valid)
            tasks.append(task)
    macros, summary, comparisons = [], [], []
    for source in (*sources, "All"):
        row = {"source": source}
        for seed in seeds:
            selected = [r for r in detail if source == "All" or r["source"] == source]
            selected = [r for r in selected if r["seed"] == seed]
            macro = {"source": source, "seed": seed}
            for method in methods:
                complete_values = [r[method] for r in selected]
                macro[method] = statistics.mean(complete_values) if complete_values and all(v is not None for v in complete_values) else None
            macros.append(macro)
        for method in methods:
            selected_values = [r[method] for r in macros if r["source"] == source and r[method] is not None]
            row[method], row[method + "_std"] = mean_std(selected_values)
            row[method + "_seeds"] = len(selected_values)
        summary.append(row)
        for question, (method, control) in comparisons_spec.items():
            deltas = [r[method] - r[control] for r in macros if r["source"] == source and r[method] is not None and r[control] is not None]
            mean, deviation = mean_std(deltas)
            comparisons.append({"question": question, "source": source, "method": method, "control": control,
                                "delta_pp": mean, "std_pp": deviation, "paired_seeds": len(deltas)})
    for name, rows in (("accuracy_per_seed", detail), ("accuracy_tasks", tasks), ("macro_per_seed", macros),
                       ("source_summary", summary), ("paired_comparisons", comparisons), ("resources", resources),
                       ("loss_curves", curves), ("pseudo_label_diagnostics", diagnostics), ("stage_refresh_events", events),
                       ("offline_pseudo_label_quality", offline_points), ("offline_bank_quality", offline_banks),
                       ("offline_teacher_quality", offline_teachers)):
        flat_csv(output / f"{name}.csv", rows)
    target_summary = []
    for target in (*DOMAINS, "All"):
        row = {"target": target}
        for method in methods:
            seed_values = []
            for seed in seeds:
                selected = [r[method] for r in detail if r["seed"] == seed and (target == "All" or r["target"] == target)]
                if selected and all(value is not None for value in selected):
                    seed_values.append(statistics.mean(selected))
            row[method], row[method + "_std"] = mean_std(seed_values)
            row[method + "_seeds"] = len(seed_values)
        target_summary.append(row)
    flat_csv(output / "target_summary.csv", target_summary)
    flat_csv(output / "actual_training_teacher_contributions.csv", actual_contributions)
    bank_by_version = {(r["variant"], r["source"], r["target"], r["seed"], r["teacher_version"]): r
                       for r in offline_banks}
    refresh_summary = []
    for event in events:
        if event["event"] not in ("admission", "refresh") or "coverage" not in event:
            continue
        key = (event["variant"], event["source"], event["target"], event["seed"], event["new_teacher_version"])
        quality = bank_by_version.get(key, {})
        refresh_summary.append({k: event.get(k) for k in ("variant", "source", "target", "seed", "step", "stage", "event")}
                               | {"teacher_version": event["new_teacher_version"],
                                  "coverage_before": event.get("previous_coverage"), "coverage_after": event["coverage"],
                                  "accepted_samples": event["accepted_samples"],
                                  "precision_before": quality.get("previous_accepted_pseudo_label_precision"),
                                  "precision_after": quality.get("accepted_pseudo_label_precision"),
                                  "precision_change": quality.get("accepted_precision_change"),
                                  "oracle_cosine": quality.get("oracle_centroid_cosine_mean"),
                                  "centroid_shift_mean": event.get("centroid_shift_mean"),
                                  "centroid_cosine_mean": event.get("centroid_cosine_mean"),
                                  "accepted_set_jaccard": event.get("accepted_set_jaccard"),
                                  "pseudo_labels_changed": event.get("pseudo_labels_changed"),
                                  "teacher_prediction_agreement": event.get("refresh_teacher_prediction_agreement"),
                                  "refresh_seconds": event["refresh_seconds"]})
    flat_csv(output / "bank_refresh_summary.csv", refresh_summary)
    accuracy_curves = []
    for point in evaluations:
        for target, result in point["targets"].items():
            accuracy_curves.append({k: point[k] for k in ("variant", "source", "seed", "step", "stage", "event", "teacher_version")}
                                   | {"target": target, "accuracy": result["accuracy"] * 100, "samples": result["samples"],
                                      "effective_updates": point["effective_updates"]})
    flat_csv(output / "accuracy_curves.csv", accuracy_curves)
    latest_diagnostics = {}
    for record in diagnostics:
        key = record["variant"], record["source"], record["target"], record["seed"]
        if key not in latest_diagnostics or record["step"] >= latest_diagnostics[key]["step"]:
            latest_diagnostics[key] = record
    clipart = [r for r in latest_diagnostics.values() if r["target"] == "Clipart"]
    flat_csv(output / "clipart_teacher_contribution.csv", clipart)
    flat_csv(output / "clipart_actual_training_contribution.csv",
             [r for r in actual_contributions if r["target"] == "Clipart"])
    hard_groups = {}
    for record in offline_points:
        key = record["variant"], record["source"], record["target"], record["seed"]
        hard_groups.setdefault(key, []).append(record)
    hard_changes = []
    for (method, source, target, seed), records in hard_groups.items():
        records.sort(key=lambda r: (r["step"], {"pre_teacher_change": 0, "post_teacher_change": 1,
                                              "interval": 2, "stage_end": 3, "final": 4}.get(r["event"], 0)))
        first, last = records[0], records[-1]
        initial_hard, final_hard = set(first["hard_classes"]), set(last["hard_classes"])
        changes = []
        for before, after in zip(first["per_class"], last["per_class"]):
            a, b = before["pseudo_label_accuracy"], after["pseudo_label_accuracy"]
            changes.append(b - a if a is not None and b is not None else None)
        hard_changes.append({"variant": method, "source": source, "target": target, "seed": seed,
                             "first_step": first["step"], "last_step": last["step"],
                             "initial_hard_classes": sorted(initial_hard), "final_hard_classes": sorted(final_hard),
                             "newly_hard_classes": sorted(final_hard - initial_hard),
                             "no_longer_hard_classes": sorted(initial_hard - final_hard),
                             "pseudo_label_accuracy_change_by_class": changes})
    flat_csv(output / "hard_class_changes.csv", hard_changes)
    candidates = []
    for question, (method, control) in comparisons_spec.items():
        if method == "R0M":
            continue
        paired = [r for r in macros if r["seed"] == 42 and r[method] is not None and r[control] is not None]
        all_row = next((r for r in paired if r["source"] == "All"), None)
        positives = sum(r[method] > r[control] for r in paired if r["source"] != "All")
        if all_row is not None and len(sources) == 4 and all_row[method] - all_row[control] >= .5 and positives >= 3:
            candidates.append({"question": question, "method": method, "control": control,
                               "seed42_macro_gain_pp": all_row[method] - all_row[control], "positive_sources": positives})
    (output / "followup_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (output / "domain_orders.json").write_text(json.dumps(orders, indent=2, ensure_ascii=False) + "\n")

    def display(row, method):
        value = row.get(method)
        if value is None:
            return "pending"
        deviation = row.get(method + "_std")
        return f"{value:.2f}" + (f" ± {deviation:.2f}" if deviation is not None else "")
    lines = ["# B 基线：跨目标域 Teacher 与 Target Bank 受控实验", "",
             f"Seeds: {list(seeds)}. 已完成 {len(runs)} 个训练结果（包含保留的 R0）；只用固定最终 checkpoint。", "",
             "R0 为原始 B；R0M 为预算对齐的联合无 Bank 对照。每个 Source/Target 的有效更新次数相同；"
             "顺序训练有三倍全局步数，Source 每三步更新一次。Target 梯度系数保持 lambda/3，学习率按各分支有效更新推进。", "",
             "## 十二个迁移任务（%）", "",
             "| Source | Target | " + " | ".join(methods) + " |",
             "|---|---|" + "---:|" * len(methods)]
    for row in tasks:
        lines.append("| " + " | ".join([row["source"], row["target"], *[display(row, m) for m in methods]]) + " |")
    lines += ["", "## 各 Source 与总体 Macro（%）", "", "| Source | " + " | ".join(methods) + " |",
              "|---|" + "---:|" * len(methods)]
    for row in summary:
        lines.append("| " + " | ".join([row["source"], *[display(row, m) for m in methods]]) + " |")
    lines += ["", "## 按 Target 汇总（%）", "", "| Target | " + " | ".join(methods) + " |",
              "|---|" + "---:|" * len(methods)]
    for row in target_summary:
        lines.append("| " + " | ".join([row["target"], *[display(row, m) for m in methods]]) + " |")
    lines += ["", "多 seed 的标准差在 seed 内先对完整任务做 macro，再跨 seed 计算样本标准差；"
              "缺少任一任务的 seed 不计入总体 macro。配对差值先在同一 seed 内计算。", "",
              "## 对四个研究问题的直接对照", "", "| 问题 | 比较 | Macro 差值（百分点） | 配对 seeds |",
              "|---|---|---:|---:|"]
    for row in comparisons:
        if row["source"] == "All":
            value = "pending" if row["delta_pp"] is None else f"{row['delta_pp']:+.2f}" + (f" ± {row['std_pp']:.2f}" if row["std_pp"] is not None else "")
            lines.append(f"| {row['question']} | {row['method']} − {row['control']} | {value} | {row['paired_seeds']} |")
    lines += ["", "R1 的差异包括 E2H 训练组织及 Source Teacher 成熟度随阶段变化。B 的 Target 参数独立，"
              "R1 本身没有跨 Target 参数传递，因此不能把该差异解释为纯粹的跨域课程知识迁移。", "",
              "## 无标签难度排序", ""]
    for source, order in orders.items():
        lines.append(f"- {source}: {' → '.join(order['order'])}; {order['rule']}。")
    lines += ["", "详细 entropy、confidence、acceptance 与排序 Teacher SHA 见 `domain_orders.json`。"
              "排序复用已有 source-only seed42 模型；Target 真标签不参与排序。", "",
              "## 伪标签与 Bank 诊断", "",
              "| Variant | 最终 SPL 平均置信度 | Acceptance | 权重熵 | 近 one-hot 类别×样本比例 | 多有效 Teacher 下的 one-hot 比例 |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in (m for m in methods if m != "R0"):
        selected = [r for r in latest_diagnostics.values() if r["variant"] == method]
        if selected:
            average = lambda key: statistics.mean(key(r) for r in selected)
            conditional = [r['contribution'].get('one_hot_multi_valid_class_sample_fraction') for r in selected]
            conditional = [v for v in conditional if v is not None]
            conditional_text = f"{statistics.mean(conditional):.3f}" if conditional else "N/A"
            lines.append(f"| {method} | {average(lambda r: r['mean_max_confidence']):.3f} | "
                         f"{average(lambda r: r['acceptance_rate']):.3f} | "
                         f"{average(lambda r: r['contribution']['weight_entropy']):.4f} | "
                         f"{average(lambda r: r['contribution']['near_one_hot_class_sample_fraction']):.3f} | {conditional_text} |")
    lines += ["", "权重诊断包含每域 min/max/mean、各类主要参考域、距离/softmax 输入分布、Invalid mask 和 Source-only 回退。"
              "单一可用 Teacher 自然为 one-hot；判断 beta 饱和时须区分实际多 Teacher 有效的类别。", "",
              "上表是训练结束时可用 ensemble 的全缓存诊断。E2H 早期已冻结 Target 不会学习后来准入的 Teacher；"
              "实际训练批次的历史 Teacher 版本与权重另见 `actual_training_teacher_contributions.csv`，"
              "由固定 feature/batch streams 与保存的 Bank 版本无标签重放计算，无优化器更新。", "",
              "离线真标签分析仅在最终训练后执行，见 `offline_*_quality.csv`：全部伪标签 accuracy、接受样本 precision、"
              "各类 precision、混淆矩阵、困难类别及 Oracle Centroid cosine。Oracle Centroid 不参与训练。"
              "阶段、准入与刷新前后指标见 `stage_refresh_events.csv` 和各 run 的完整快照。", "",
              "## 多 seed 后续运行条件", "",
              "预先约定 seed42 相对规定对照 ≥0.5 个 macro 百分点且至少三个 Source 改善，才追加 seeds41/43。"
              "这只是算力分配条件，不能替代统计显著性检验。", "",
              f"满足条件的比较: {json.dumps(candidates, ensure_ascii=False)}", "",
              "## 复现与限制", "",
              "运行命令、冻结权重 SHA、原始训练代码 SHA、新代码 SHA、缓存标识与预算计数保存在各 run 的 config/environment/last.pt。"
              "原始 R0 只存最终 accuracy，无法补造其历史 accuracy/伪标签曲线；R0M 提供相同预算下的完整曲线。"
              "新的 elapsed 包含训练中固定诊断、checkpoint 和最终离线分析，与旧 R0 的纯训练时间口径不同。", "",
              "完整诊断大数组保留在 runs；报告目录保存表格。只有 Source CE 与 SPL Soft CE 两项损失。"
              "本轮不包含 OT、对比学习、额外蒸馏、Replay、Feature Alignment 或 Entropy Minimization。", ""]
    (output / "report.md").write_text("\n".join(lines))
    complete = all(row[method + "_seeds"] == len(seeds) for row in summary for method in methods)
    print(json.dumps({"complete": complete, "runs": len(runs), "followup_candidates": candidates}))
    return {"complete": complete, "summary": summary, "comparisons": comparisons, "candidates": candidates}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/b_reliability")
    cli.add_argument("--r0-dir", default="runs/mtda_spl")
    cli.add_argument("--output-dir", default="reports/b_reliability")
    cli.add_argument("--seeds", nargs="+", type=int, default=[42])
    cli.add_argument("--sources", nargs="+", default=list(DOMAINS))
    cli.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                     help="report only the methods requested for multi-seed follow-up")
    args = cli.parse_args()
    summarize(args.runs_dir, args.r0_dir, args.output_dir, tuple(args.seeds), tuple(args.sources), tuple(args.methods))


if __name__ == "__main__":
    main()
