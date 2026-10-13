# B 基线：跨目标域 Teacher 与 Target Bank 受控实验

Seeds: [41, 42, 43]. 已完成 60 个训练结果（包含保留的 R0）；只用固定最终 checkpoint。

R0 为原始 B；R0M 为预算对齐的联合无 Bank 对照。每个 Source/Target 的有效更新次数相同；顺序训练有三倍全局步数，Source 每三步更新一次。Target 梯度系数保持 lambda/3，学习率按各分支有效更新推进。

## 十二个迁移任务（%）

| Source | Target | R0 | R0M | R1 | R2 | R4 |
|---|---|---:|---:|---:|---:|---:|
| Art | Clipart | 52.29 ± 0.45 | 52.29 ± 0.45 | 52.43 ± 0.41 | 54.39 ± 0.26 | 54.39 ± 0.13 |
| Art | Product | 81.08 ± 0.32 | 81.08 ± 0.32 | 81.07 ± 0.23 | 83.02 ± 0.35 | 83.65 ± 0.22 |
| Art | Real World | 84.09 ± 0.21 | 84.09 ± 0.21 | 84.85 ± 0.39 | 84.85 ± 0.39 | 85.23 ± 0.17 |
| Clipart | Art | 71.80 ± 0.67 | 71.80 ± 0.67 | 71.61 ± 0.46 | 74.70 ± 0.18 | 74.67 ± 0.27 |
| Clipart | Product | 81.74 ± 0.03 | 81.74 ± 0.03 | 83.32 ± 0.50 | 83.32 ± 0.50 | 84.65 ± 0.48 |
| Clipart | Real World | 82.61 ± 0.48 | 82.61 ± 0.48 | 83.15 ± 0.13 | 84.91 ± 0.23 | 84.91 ± 0.27 |
| Product | Art | 70.55 ± 0.63 | 70.55 ± 0.63 | 71.20 ± 0.30 | 73.37 ± 0.31 | 73.58 ± 0.05 |
| Product | Clipart | 52.33 ± 0.71 | 52.33 ± 0.71 | 52.21 ± 0.44 | 54.33 ± 0.45 | 54.14 ± 0.48 |
| Product | Real World | 84.03 ± 0.31 | 84.03 ± 0.31 | 84.95 ± 0.22 | 84.95 ± 0.22 | 84.94 ± 0.17 |
| Real World | Art | 73.81 ± 0.42 | 73.81 ± 0.42 | 74.18 ± 0.31 | 74.70 ± 0.19 | 74.76 ± 0.12 |
| Real World | Clipart | 53.65 ± 0.12 | 53.65 ± 0.12 | 53.63 ± 0.22 | 54.54 ± 0.22 | 54.78 ± 0.34 |
| Real World | Product | 86.06 ± 0.14 | 86.06 ± 0.14 | 86.50 ± 0.15 | 86.50 ± 0.15 | 85.79 ± 0.13 |

## 各 Source 与总体 Macro（%）

| Source | R0 | R0M | R1 | R2 | R4 |
|---|---:|---:|---:|---:|---:|
| Art | 72.49 ± 0.06 | 72.49 ± 0.06 | 72.78 ± 0.07 | 74.09 ± 0.13 | 74.42 ± 0.09 |
| Clipart | 78.72 ± 0.38 | 78.72 ± 0.38 | 79.36 ± 0.19 | 80.98 ± 0.03 | 81.41 ± 0.14 |
| Product | 68.97 ± 0.24 | 68.97 ± 0.24 | 69.45 ± 0.11 | 70.88 ± 0.16 | 70.88 ± 0.17 |
| Real World | 71.17 ± 0.16 | 71.17 ± 0.16 | 71.44 ± 0.17 | 71.91 ± 0.03 | 71.77 ± 0.17 |
| All | 72.84 ± 0.15 | 72.84 ± 0.15 | 73.26 ± 0.05 | 74.47 ± 0.03 | 74.62 ± 0.05 |

## 按 Target 汇总（%）

| Target | R0 | R0M | R1 | R2 | R4 |
|---|---:|---:|---:|---:|---:|
| Art | 72.06 ± 0.51 | 72.06 ± 0.51 | 72.33 ± 0.18 | 74.26 ± 0.10 | 74.34 ± 0.12 |
| Clipart | 52.76 ± 0.28 | 52.76 ± 0.28 | 52.76 ± 0.18 | 54.42 ± 0.24 | 54.43 ± 0.18 |
| Product | 82.96 ± 0.15 | 82.96 ± 0.15 | 83.63 ± 0.07 | 84.28 ± 0.10 | 84.70 ± 0.07 |
| Real World | 83.57 ± 0.22 | 83.57 ± 0.22 | 84.32 ± 0.19 | 84.90 ± 0.13 | 85.03 ± 0.12 |
| All | 72.84 ± 0.15 | 72.84 ± 0.15 | 73.26 ± 0.05 | 74.47 ± 0.03 | 74.62 ± 0.05 |

多 seed 的标准差在 seed 内先对完整任务做 macro，再跨 seed 计算样本标准差；缺少任一任务的 seed 不计入总体 macro。配对差值先在同一 seed 内计算。

## 对四个研究问题的直接对照

| 问题 | 比较 | Macro 差值（百分点） | 配对 seeds |
|---|---|---:|---:|
| E2H | R1 − R0M | +0.42 ± 0.18 | 3 |
| E2H_uniform | R2 − R1 | +1.21 ± 0.08 | 3 |
| joint_uniform | R4 − R0M | +1.79 ± 0.16 | 3 |
| matched_control | R0M − R0 | +0.00 ± 0.00 | 3 |

R1 的差异包括 E2H 训练组织及 Source Teacher 成熟度随阶段变化。B 的 Target 参数独立，R1 本身没有跨 Target 参数传递，因此不能把该差异解释为纯粹的跨域课程知识迁移。

## 无标签难度排序

- Art: Real World → Product → Clipart; entropy ascending; ties: acceptance/max-confidence descending, name。
- Clipart: Product → Real World → Art; entropy ascending; ties: acceptance/max-confidence descending, name。
- Product: Real World → Art → Clipart; entropy ascending; ties: acceptance/max-confidence descending, name。
- Real World: Product → Art → Clipart; entropy ascending; ties: acceptance/max-confidence descending, name。

详细 entropy、confidence、acceptance 与排序 Teacher SHA 见 `domain_orders.json`。排序复用已有 source-only seed42 模型；Target 真标签不参与排序。

## 伪标签与 Bank 诊断

| Variant | 最终 SPL 平均置信度 | Acceptance | 权重熵 | 近 one-hot 类别×样本比例 | 多有效 Teacher 下的 one-hot 比例 |
|---|---:|---:|---:|---:|---:|
| R0M | 0.754 | 0.641 | 0.0000 | 1.000 | N/A |
| R1 | 0.754 | 0.641 | 0.0000 | 1.000 | N/A |
| R2 | 0.729 | 0.612 | 1.0986 | 0.000 | 0.000 |
| R4 | 0.727 | 0.609 | 1.0986 | 0.000 | 0.000 |

权重诊断包含每域 min/max/mean、各类主要参考域、距离/softmax 输入分布、Invalid mask 和 Source-only 回退。单一可用 Teacher 自然为 one-hot；判断 beta 饱和时须区分实际多 Teacher 有效的类别。

上表是训练结束时可用 ensemble 的全缓存诊断。E2H 早期已冻结 Target 不会学习后来准入的 Teacher；实际训练批次的历史 Teacher 版本与权重另见 `actual_training_teacher_contributions.csv`，由固定 feature/batch streams 与保存的 Bank 版本无标签重放计算，无优化器更新。

离线真标签分析仅在最终训练后执行，见 `offline_*_quality.csv`：全部伪标签 accuracy、接受样本 precision、各类 precision、混淆矩阵、困难类别及 Oracle Centroid cosine。Oracle Centroid 不参与训练。阶段、准入与刷新前后指标见 `stage_refresh_events.csv` 和各 run 的完整快照。

## 多 seed 后续运行条件

预先约定 seed42 相对规定对照 ≥0.5 个 macro 百分点且至少三个 Source 改善，才追加 seeds41/43。这只是算力分配条件，不能替代统计显著性检验。

满足条件的比较: [{"question": "E2H_uniform", "method": "R2", "control": "R1", "seed42_macro_gain_pp": 1.2868367742821505, "positive_sources": 4}, {"question": "joint_uniform", "method": "R4", "control": "R0M", "seed42_macro_gain_pp": 1.6483354245224433, "positive_sources": 4}]

## 复现与限制

运行命令、冻结权重 SHA、原始训练代码 SHA、新代码 SHA、缓存标识与预算计数保存在各 run 的 config/environment/last.pt。原始 R0 只存最终 accuracy，无法补造其历史 accuracy/伪标签曲线；R0M 提供相同预算下的完整曲线。新的 elapsed 包含训练中固定诊断、checkpoint 和最终离线分析，与旧 R0 的纯训练时间口径不同。

完整诊断大数组保留在 runs；报告目录保存表格。只有 Source CE 与 SPL Soft CE 两项损失。本轮不包含 OT、对比学习、额外蒸馏、Replay、Feature Alignment 或 Entropy Minimization。
