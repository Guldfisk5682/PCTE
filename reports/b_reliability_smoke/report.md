# B 基线：跨目标域 Teacher 与 Target Bank 受控实验

Seeds: [42]. 已完成 8 个训练结果（包含保留的 R0）；只用固定最终 checkpoint。

R0 为原始 B；R0M 为预算对齐的联合无 Bank 对照。每个 Source/Target 的有效更新次数相同；顺序训练有三倍全局步数，Source 每三步更新一次。Target 梯度系数保持 lambda/3，学习率按各分支有效更新推进。

## 十二个迁移任务（%）

| Source | Target | R0 | R0M | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Art | Clipart | 50.88 | 50.88 | 51.20 | 51.23 | 51.16 | 50.56 | 50.97 | 50.86 |
| Art | Product | 80.85 | 80.85 | 80.72 | 81.62 | 81.93 | 81.32 | 81.26 | 81.91 |
| Art | Real World | 80.67 | 80.67 | 80.24 | 80.24 | 80.24 | 81.16 | 80.95 | 80.81 |

## 各 Source 与总体 Macro（%）

| Source | R0 | R0M | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Art | 70.80 | 70.80 | 70.72 | 71.03 | 71.11 | 71.01 | 71.06 | 71.19 |
| All | 70.80 | 70.80 | 70.72 | 71.03 | 71.11 | 71.01 | 71.06 | 71.19 |

多 seed 的标准差在 seed 内先对完整任务做 macro，再跨 seed 计算样本标准差；缺少任一任务的 seed 不计入总体 macro。配对差值先在同一 seed 内计算。

## 对四个研究问题的直接对照

| 问题 | 比较 | Macro 差值（百分点） | 配对 seeds |
|---|---|---:|---:|
| E2H | R1 − R0M | -0.08 | 1 |
| E2H_uniform | R2 − R1 | +0.31 | 1 |
| E2H_bank | R3 − R2 | +0.08 | 1 |
| joint_uniform | R4 − R0M | +0.21 | 1 |
| joint_bank | R5 − R4 | +0.05 | 1 |
| refresh | R6 − R5 | +0.13 | 1 |
| matched_control | R0M − R0 | +0.00 | 1 |

R1 的差异包括 E2H 训练组织及 Source Teacher 成熟度随阶段变化。B 的 Target 参数独立，R1 本身没有跨 Target 参数传递，因此不能把该差异解释为纯粹的跨域课程知识迁移。

## 无标签难度排序

- Art: Real World → Product → Clipart; entropy ascending; ties: acceptance/max-confidence descending, name。

详细 entropy、confidence、acceptance 与排序 Teacher SHA 见 `domain_orders.json`。排序复用已有 source-only seed42 模型；Target 真标签不参与排序。

## 伪标签与 Bank 诊断

| Variant | 最终 SPL 平均置信度 | Acceptance | 权重熵 | 近 one-hot 类别×样本比例 |
|---|---:|---:|---:|---:|
| R0M | 0.701 | 0.575 | 0.0000 | 1.000 |
| R1 | 0.701 | 0.575 | 0.0000 | 1.000 |
| R2 | 0.691 | 0.566 | 1.0986 | 0.000 |
| R3 | 0.694 | 0.568 | 0.3200 | 0.296 |
| R4 | 0.689 | 0.562 | 1.0986 | 0.000 |
| R5 | 0.688 | 0.558 | 0.3125 | 0.308 |
| R6 | 0.696 | 0.572 | 0.3207 | 0.293 |

权重诊断包含每域 min/max/mean、各类主要参考域、距离/softmax 输入分布、Invalid mask 和 Source-only 回退。单一可用 Teacher 自然为 one-hot；判断 beta 饱和时须区分实际多 Teacher 有效的类别。

离线真标签分析仅在最终训练后执行，见 `offline_*_quality.csv`：全部伪标签 accuracy、接受样本 precision、各类 precision、混淆矩阵、困难类别及 Oracle Centroid cosine。Oracle Centroid 不参与训练。阶段、准入与刷新前后指标见 `stage_refresh_events.csv` 和各 run 的完整快照。

## 多 seed 后续运行条件

预先约定 seed42 相对规定对照 ≥0.5 个 macro 百分点且至少三个 Source 改善，才追加 seeds41/43。这只是算力分配条件，不能替代统计显著性检验。

满足条件的比较: []

## 复现与限制

运行命令、冻结权重 SHA、原始训练代码 SHA、新代码 SHA、缓存标识与预算计数保存在各 run 的 config/environment/last.pt。原始 R0 只存最终 accuracy，无法补造其历史 accuracy/伪标签曲线；R0M 提供相同预算下的完整曲线。新的 elapsed 包含训练中固定诊断、checkpoint 和最终离线分析，与旧 R0 的纯训练时间口径不同。

完整诊断大数组保留在 runs；报告目录保存表格。只有 Source CE 与 SPL Soft CE 两项损失。本轮不包含 OT、对比学习、额外蒸馏、Replay、Feature Alignment 或 Entropy Minimization。
