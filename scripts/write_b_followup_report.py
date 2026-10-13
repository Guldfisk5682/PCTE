#!/usr/bin/env python3
"""Render the final controlled study without choosing training checkpoints."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.summarize_b_followup import summarize, DOMAINS, average


def render(runs, baseline, output, agreement_dir):
    macro, paired, weights, candidates = summarize(runs, baseline, output)
    values = {(r["method"], r["seed"]): r["macro_accuracy"] for r in macro}
    agreement = json.loads((agreement_dir / "agreement.json").read_text())
    final = [r for r in agreement if r["event"] == "final"]
    loo = json.loads((agreement_dir / "teacher_leave_one_out.json").read_text())
    lines = ["# R4 后续受控实验报告", "", "## 1. 多随机种子稳定性", "",
             "沿用固定最终 Checkpoint，按 12 个 Source→Target 任务等权求 Macro，再计算三个 seed 的均值与样本标准差。",
             "seeds 41/43 已在前一轮完成，本轮复用并审核；没有重新训练或按目标准确率挑选 Checkpoint。", "",
             "| Seed | B / R0M | R2 | R4 | R4 − B（pp） |", "|---|---:|---:|---:|---:|"]
    for seed in (41, 42, 43):
        lines.append(f"| {seed} | {values['R0M',seed]:.4f} | {values['R2',seed]:.4f} | {values['R4',seed]:.4f} | {values['R4',seed]-values['R0M',seed]:+.4f} |")
    for method in ("R2", "R4"):
        comparison = next(r for r in paired if r["method"] == method)
        acc = [values[method,s] for s in (41,42,43)]
        lines += ["", f"{method}: **{average(acc):.4f} ± {statistics.stdev(acc):.4f}%**；相对 B 的配对增益为 "
                  f"**{comparison['mean_gain_pp']:+.4f} ± {comparison['std_gain_pp']:.4f} pp**。"]
    lines += ["", "## 2. R4 Teacher Agreement 与错误归因", "",
              "以下是 Teacher 伪标签准确率，不是最终 Student 准确率。所有真实目标标签仅用于训练完成后的离线分析。",
              "投票一致定义为：原始 Base+Source SPL 的预测，与当时已准入且不属于当前目标域的每个 Target Prompt 预测相同。",
              "它不是 Source Prompt 单独预测的三教师多数投票，训练仍在相似度层面融合。", "",
              "| 最终 Teacher 指标（三 seed） | 所有任务 | Clipart 作为目标域 |", "|---|---:|---:|"]
    for name,key in (("原始 SPL Accuracy","base_accuracy"),("R4 Cross-target Teacher Accuracy","cross_accuracy"),
                     ("原始错误被纠正 / 全部样本","correction_rate"),("原始正确变错 / 全部样本","damage_rate"),
                     ("双方相同错误 / 全部样本","same_wrong_rate"),("投票一致且错误 / 全部样本","unanimous_wrong_rate"),
                     ("投票与 Cross Teacher 同为相同错误 / 全部样本","unanimous_and_cross_same_wrong_rate"),
                     ("投票共同错误但 Cross 融合纠正 / 全部样本","unanimous_but_cross_correct_rate"),
                     ("置信度 ≥0.7 且投票一致错误 / 全部样本","accepted_unanimous_wrong_rate"),
                     ("投票一致样本的 Precision","unanimous_vote_precision"),
                     ("原始 SPL 与 Cross Teacher 一致率","base_cross_agreement")):
        all_value = 100*average([r[key] for r in final])
        clip_value = 100*average([r[key] for r in final if r["target"] == "Clipart"])
        lines.append(f"| {name} | {all_value:.3f}% | {clip_value:.3f}% |")
    lines += ["", "| 准入节点 | Cross Teacher 准入前 | 准入后 | 即时变化（pp） |", "|---|---:|---:|---:|"]
    for step in (250,500,750):
        pre = average([r["cross_accuracy"] for r in agreement if r["step"] == step and r["event"] == "pre_teacher_change"])*100
        post = average([r["cross_accuracy"] for r in agreement if r["step"] == step and r["event"] == "post_teacher_change"])*100
        lines.append(f"| {step} | {pre:.3f}% | {post:.3f}% | {post-pre:+.3f} |")
    actual_vote_rows = [r for r in final if "actual_prompt_teacher_agreement_rate" in r]
    if actual_vote_rows:
        lines += ["", "实际参考 Prompt 的一致率另外计算：Source Prompt + 两个其他 Target Prompt，排除固定 Base CLIP。", "",
                  "| 最终实际 Prompt 指标 | 所有任务 | Clipart 作为目标域 |", "|---|---:|---:|"]
        for name,key in (("三 Prompt 一致率","actual_prompt_teacher_agreement_rate"),
                         ("三 Prompt 一致且错误 / 全样本","actual_prompt_teacher_unanimous_wrong_rate"),
                         ("三 Prompt 一致且 Cross 同样错误 / 全样本","actual_prompt_teacher_unanimous_wrong_cross_rate"),
                         ("三 Prompt 一致样本 Precision","actual_prompt_teacher_unanimous_precision")):
            all_value=100*average([r[key] for r in actual_vote_rows])
            clip_value=100*average([r[key] for r in actual_vote_rows if r["target"]=="Clipart"])
            lines.append(f"| {name} | {all_value:.3f}% | {clip_value:.3f}% |")
    lines += ["", "准入前后使用同一个 Student 与 Source 状态，因此上表衡量教师加入的即时伪标签效果，不是 Student 训练收益。",
              "第一次准入作用最大，后续收益递减。Clipart 在所有 Source 下均排最后：自身准入不改变其 SPL，验证了自身教师排除规则。",
              "Clipart 被纠正的样本中，约 91%–93% 至少有一个跨域 Teacher 的首选类别正确；这属于支持证据，不能直接视为因果贡献。",
              "Clipart 的 Computer、Sink、Marker、Mop 等类别仍有明显共同错误，单纯提高投票一致率无法处理这些错误。", ""]
    if loo:
        analysis_protocol = json.loads((agreement_dir / "protocol.json").read_text())
        source_description = ("加载原 R4 Source 最终 Checkpoint 的完整 Prompt 参数进行 CPU 推理，不执行任何训练更新。"
                              if analysis_protocol.get("source_final_checkpoint_inference_only") else
                              "重建独立 Source CE 轨迹，最终 Source 参数逐项等于原 R4。")
        lines += ["### 固定教师的 Leave-one-out 贡献", "",
                  source_description + "移除一个 Target Teacher 后重新融合其余相似度；不重训 Student。",
                  "下表为 Teacher 伪标签的边际准确率变化，正值表示加入该 Teacher 有益。", "",
                  "| Source | 当前 Target | 被移除 Teacher | 加入该教师的净贡献（pp） |", "|---|---|---|---:|"]
        grouped = defaultdict(list)
        for r in loo:
            if r["event"] == "final":
                grouped[r["source"],r["target"],r["removed_teacher"]].append(r["teacher_marginal_gain"]*100)
        for key, ds in sorted(grouped.items()):
            lines.append(f"| {' | '.join(key)} | {average(ds):+.3f} |")
        disagreements = sum(r.get("cpu_recomputed_prediction_disagreements",0) for r in agreement)
        lines += ["", f"CPU 重算与已保存 GPU Cross Teacher 预测共有 {disagreements} 个样本点差异；错误分区始终使用原始保存预测。", ""]
    lines += ["## 3. 固定 R4 教师的 β 扫描", "",
              "以下第一轮使用 seed 42。`FUniform` 复用原 R4 的三个教师快照，按原 250/500/750 时刻开放，必须逐项复现 R4 参数。",
              "β 扫描共享同样的 Snapshot、Bank、样本流、更新预算与准入时间，只改变距离 Softmax 的 β。",
              "β=0 仍保留 Invalid Centroid Mask；因此与普通 Uniform 的差异可包含缺失类别 Mask 的影响。", "",
              "| 方法 | 12-task Macro | 相对 R4（pp） | 最终多有效域 One-hot 比例 | Clipart 的 Source 权重 |",
              "|---|---:|---:|---:|---:|"]
    for method in ("R4","R5","FUniform","FBeta0","FBeta0p1","FBeta1","FBeta10"):
        if (method,42) not in values:
            continue
        w = next((r for r in weights if r["method"]==method and r["target_group"]=="all"),None)
        c = next((r for r in weights if r["method"]==method and r["target_group"]=="Clipart"),None)
        hot = f"{100*w['final_one_hot_rate']:.2f}%" if w and w['final_one_hot_rate'] is not None else "—"
        source = f"{100*c['final_source_weight']:.2f}%" if c else "—"
        lines.append(f"| {method} | {values[method,42]:.4f}% | {values[method,42]-values['R4',42]:+.4f} | {hot} | {source} |")
    lines += ["", "One-hot 使用有至少两个有效质心的样本×类别位置，阈值为最大权重 >0.99；排除只有 Source 可用的平凡单教师位置。",
              "Source 权重是在固定 Base 系数 0.5 之外的参考教师半边内计算。实际训练批次统计另见 `actual_teacher_contributions.csv`。",
              "原 R5 的后续快照受其自身训练影响，不能将 R5 与新 β 扫描的差异全部归因于温度。", "",
              "### 几何距离与 Teacher 能力的关系", "",
              "`geometry_mismatch.csv` 使用目标真标签做离线诊断，绝不反馈给 Bank。考察真实类别位置的教师权重，",
              "并统计这些权重分给首选类别正确教师的总量。该总量不是经过校准的正确概率，也不等于融合后的准确率。", "",
              "## 4. Progressive Admission 对照", "",
              "| 方法 | Seed 42 Macro | 相对 R4（pp） | Teacher 来源与时机 |", "|---|---:|---:|---|"]
    for method, description in (("R4","在线、逐步生成并准入"),("FUniform","同样 R4 快照、原时刻准入"),
                                ("FStartInit","Step 0 全部初始未训练 Prompt 快照，随后固定"),
                                ("FStartRef","Step 0 全部已训练 R4 快照，离线时机对照")):
        if (method,42) in values:
            lines.append(f"| {method} | {values[method,42]:.4f}% | {values[method,42]-values['R4',42]:+.4f} | {description} |")
    lines += ["", "`FStartInit` 同时改变教师成熟度与准入时机；它单独不能证明渐进排序必要。",
              "`FStartRef` 与 `FUniform` 教师质量和身份完全相同，因此能隔离开放时间；但使用未来教师快照，不能作为因果在线算法。",
              "所有 Student 的 Source 与每个 Target 各 1,000 次更新。参考快照的共同 R4 准备另需每分支 1,000 次更新，",
              "由这些离线对照共享；与原 R4 比较时不能声称端到端计算预算相同。", "",
              "### Seed 42 的 12 个任务准确率", ""]
    task_rows = []
    import csv
    with (output / "accuracy_tasks.csv").open() as handle:
        task_rows = list(csv.DictReader(handle))
    task_values = {(r["method"],r["source"],r["target"]):float(r["accuracy"]) for r in task_rows if int(r["seed"])==42}
    table_methods = [m for m in ("R4","FBeta0","FBeta0p1","FBeta1","FBeta10","FStartInit","FStartRef") if (m,42) in values]
    lines += ["| Source → Target | " + " | ".join(table_methods) + " |", "|---|"+"---:|"*len(table_methods)]
    for source in DOMAINS:
        for target in DOMAINS:
            if source != target:
                lines.append(f"| {source} → {target} | " + " | ".join(f"{task_values[m,source,target]:.3f}" for m in table_methods) + " |")
    lines += ["",
              "## 5. 执行状态、证据与限制", "",
              "保留原 A/B/C 和 R0–R6。新增入口为 `train_b_followup.py`，运行树为 `runs/b_followup/`。",
              "完整任务准确率见 `accuracy_tasks.csv`，Source 表见 `source_summary.csv`，配对 seed 统计见 `paired_comparisons.csv`。",
              "错误分区、类别分区和教师边际贡献见 `agreement/`；预算、参考快照一致性与原 R4 精确复现见 `validation.json`。",
              "不新增训练损失、过滤、阈值调节或特征变换；目标标签不参与训练、准入、Bank 或 Checkpoint 选择。", "",
              "预注册追加 seed 条件：相对 R4 ≥0.5 pp 且至少三个 Source 正收益。第一轮候选如下：", ""]
    for r in candidates:
        lines.append(f"- {r['method']}: {r['gain_pp']:+.4f} pp，{r['positive_sources']}/4 Source 正收益，追加 seed 条件{'满足' if r['qualifies_multiseed'] else '不满足'}。")
    if (output / "geometry_mismatch.json").exists():
        geometry = json.loads((output / "geometry_mismatch.json").read_text())
        lines += ["", "## 6. Clipart 的类别条件几何诊断", "",
                  "| β | Source 在真实类别的权重 | Source 在其他类别的平均权重 | 真实类别的正确教师投票权重总量 |", "|---|---:|---:|---:|"]
        for beta in (0.,.1,1.,10.):
            selected = [r for r in geometry if r["target"]=="Clipart" and r["beta"]==beta]
            fields = ("source_weight_on_true_class","source_weight_on_wrong_classes","weighted_correct_teacher_vote_mass_at_true_class")
            lines.append(f"| {beta:g} | " + " | ".join(f"{100*average([r[k] for r in selected]):.3f}%" for k in fields) + " |")
        lines += ["", "这里的教师集合、Bank 与 Source 最终 Prompt 固定，因此仅度量距离权重变化。",
                  "真实类别的正确投票权重增加，并不保证整个 65 类融合分类器改进；其他类别的权重和教师相似度也影响决策。",
                  "扫描 β 同时改变集中程度及教师偏好，不能把全部变化唯一归因于 One-hot 本身。"]
    if (output / "validation.json").exists():
        audit = json.loads((output / "validation.json").read_text())
        lines += ["", f"已完成并审核 {audit['completed_runs']} 个新增 run；FUniform 全 Prompt 与 R4 一致，扫描 Bank 逐项一致，Source 轨迹和预算保持一致。"]
    (output / "report.md").write_text("\n".join(lines)+"\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs-dir", default="runs/b_followup")
    cli.add_argument("--baseline-dir", default="runs/b_reliability")
    cli.add_argument("--output-dir", default="reports/b_followup")
    cli.add_argument("--agreement-dir", default="reports/b_followup/agreement")
    args = cli.parse_args()
    render(Path(args.runs_dir),Path(args.baseline_dir),Path(args.output_dir),Path(args.agreement_dir))


if __name__ == "__main__":
    main()
