# B 基线：跨目标域 Teacher 与 Target Bank 受控实验

Seeds: [42]. 已完成 32 个训练结果（包含保留的 R0）；只用固定最终 checkpoint。

R0 为原始 B；R0M 为预算对齐的联合无 Bank 对照。每个 Source/Target 的有效更新次数相同；顺序训练有三倍全局步数，Source 每三步更新一次。Target 梯度系数保持 lambda/3，学习率按各分支有效更新推进。

## 十二个迁移任务（%）

| Source | Target | R0 | R0M | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Art | Clipart | 52.67 | 52.67 | 52.71 | 54.66 | 53.72 | 54.52 | 54.09 | 54.07 |
| Art | Product | 81.17 | 81.17 | 81.32 | 83.35 | 84.28 | 83.89 | 83.64 | 83.58 |
| Art | Real World | 83.84 | 83.84 | 84.42 | 84.42 | 84.42 | 85.10 | 84.85 | 84.76 |
| Clipart | Art | 72.23 | 72.23 | 71.86 | 74.91 | 72.39 | 74.41 | 73.09 | 72.56 |
| Clipart | Product | 81.71 | 81.71 | 82.74 | 82.74 | 82.74 | 84.12 | 84.16 | 83.60 |
| Clipart | Real World | 82.76 | 82.76 | 83.22 | 85.17 | 84.12 | 85.22 | 84.55 | 83.98 |
| Product | Art | 71.28 | 71.28 | 71.53 | 73.71 | 73.75 | 73.55 | 73.01 | 73.42 |
| Product | Clipart | 51.96 | 51.96 | 51.98 | 54.57 | 53.75 | 53.93 | 53.38 | 53.20 |
| Product | Real World | 84.32 | 84.32 | 84.78 | 84.78 | 84.78 | 85.13 | 85.10 | 84.67 |
| Real World | Art | 74.29 | 74.29 | 73.88 | 74.50 | 74.21 | 74.82 | 74.08 | 74.00 |
| Real World | Clipart | 53.56 | 53.56 | 53.52 | 54.59 | 55.10 | 55.17 | 55.12 | 55.37 |
| Real World | Product | 86.19 | 86.19 | 86.64 | 86.64 | 86.64 | 85.90 | 86.12 | 86.15 |

## 各 Source 与总体 Macro（%）

| Source | R0 | R0M | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Art | 72.56 | 72.56 | 72.82 | 74.14 | 74.14 | 74.51 | 74.20 | 74.13 |
| Clipart | 78.90 | 78.90 | 79.27 | 80.94 | 79.75 | 81.25 | 80.60 | 80.05 |
| Product | 69.19 | 69.19 | 69.43 | 71.02 | 70.76 | 70.87 | 70.50 | 70.43 |
| Real World | 71.35 | 71.35 | 71.34 | 71.91 | 71.98 | 71.96 | 71.78 | 71.84 |
| All | 73.00 | 73.00 | 73.22 | 74.50 | 74.16 | 74.65 | 74.27 | 74.11 |

## 按 Target 汇总（%）

| Target | R0 | R0M | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Art | 72.60 | 72.60 | 72.42 | 74.37 | 73.45 | 74.26 | 73.40 | 73.33 |
| Clipart | 52.73 | 52.73 | 52.74 | 54.61 | 54.19 | 54.54 | 54.20 | 54.21 |
| Product | 83.02 | 83.02 | 83.57 | 84.25 | 84.55 | 84.64 | 84.64 | 84.44 |
| Real World | 83.64 | 83.64 | 84.14 | 84.79 | 84.44 | 85.15 | 84.84 | 84.47 |
| All | 73.00 | 73.00 | 73.22 | 74.50 | 74.16 | 74.65 | 74.27 | 74.11 |

多 seed 的标准差在 seed 内先对完整任务做 macro，再跨 seed 计算样本标准差；缺少任一任务的 seed 不计入总体 macro。配对差值先在同一 seed 内计算。

## 对四个研究问题的直接对照

| 问题 | 比较 | Macro 差值（百分点） | 配对 seeds |
|---|---|---:|---:|
| E2H | R1 − R0M | +0.22 | 1 |
| E2H_uniform | R2 − R1 | +1.29 | 1 |
| E2H_bank | R3 − R2 | -0.35 | 1 |
| joint_uniform | R4 − R0M | +1.65 | 1 |
| joint_bank | R5 − R4 | -0.38 | 1 |
| refresh | R6 − R5 | -0.16 | 1 |
| matched_control | R0M − R0 | +0.00 | 1 |

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
| R0M | 0.754 | 0.642 | 0.0000 | 1.000 | N/A |
| R1 | 0.754 | 0.642 | 0.0000 | 1.000 | N/A |
| R2 | 0.730 | 0.613 | 1.0986 | 0.000 | 0.000 |
| R3 | 0.739 | 0.625 | 0.3428 | 0.272 | 0.272 |
| R4 | 0.728 | 0.610 | 1.0986 | 0.000 | 0.000 |
| R5 | 0.736 | 0.620 | 0.3401 | 0.277 | 0.277 |
| R6 | 0.735 | 0.619 | 0.3364 | 0.282 | 0.281 |

权重诊断包含每域 min/max/mean、各类主要参考域、距离/softmax 输入分布、Invalid mask 和 Source-only 回退。单一可用 Teacher 自然为 one-hot；判断 beta 饱和时须区分实际多 Teacher 有效的类别。

上表是训练结束时可用 ensemble 的全缓存诊断。E2H 早期已冻结 Target 不会学习后来准入的 Teacher；实际训练批次的历史 Teacher 版本与权重另见 `actual_training_teacher_contributions.csv`，由固定 feature/batch streams 与保存的 Bank 版本无标签重放计算，无优化器更新。

离线真标签分析仅在最终训练后执行，见 `offline_*_quality.csv`：全部伪标签 accuracy、接受样本 precision、各类 precision、混淆矩阵、困难类别及 Oracle Centroid cosine。Oracle Centroid 不参与训练。阶段、准入与刷新前后指标见 `stage_refresh_events.csv` 和各 run 的完整快照。

## 多 seed 后续运行条件

预先约定 seed42 相对规定对照 ≥0.5 个 macro 百分点且至少三个 Source 改善，才追加 seeds41/43。这只是算力分配条件，不能替代统计显著性检验。

满足条件的比较: [{"question": "E2H_uniform", "method": "R2", "control": "R1", "seed42_macro_gain_pp": 1.2868367742821505, "positive_sources": 4}, {"question": "joint_uniform", "method": "R4", "control": "R0M", "seed42_macro_gain_pp": 1.6483354245224433, "positive_sources": 4}]

## 复现与限制

运行命令、冻结权重 SHA、原始训练代码 SHA、新代码 SHA、缓存标识与预算计数保存在各 run 的 config/environment/last.pt。原始 R0 只存最终 accuracy，无法补造其历史 accuracy/伪标签曲线；R0M 提供相同预算下的完整曲线。新的 elapsed 包含训练中固定诊断、checkpoint 和最终离线分析，与旧 R0 的纯训练时间口径不同。

完整诊断大数组保留在 runs；报告目录保存表格。只有 Source CE 与 SPL Soft CE 两项损失。本轮不包含 OT、对比学习、额外蒸馏、Replay、Feature Alignment 或 Entropy Minimization。
