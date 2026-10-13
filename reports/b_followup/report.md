# R4 后续受控实验报告

## 1. 多随机种子稳定性

沿用固定最终 Checkpoint，按 12 个 Source→Target 任务等权求 Macro，再计算三个 seed 的均值与样本标准差。
seeds 41/43 已在前一轮完成，本轮复用并审核；没有重新训练或按目标准确率挑选 Checkpoint。

| Seed | B / R0M | R2 | R4 | R4 − B（pp） |
|---|---:|---:|---:|---:|
| 41 | 72.8177 | 74.4404 | 74.5697 | +1.7519 |
| 42 | 72.9987 | 74.5042 | 74.6470 | +1.6483 |
| 43 | 72.6935 | 74.4513 | 74.6505 | +1.9570 |

R2: **74.4653 ± 0.0341%**；相对 B 的配对增益为 **+1.6286 ± 0.1263 pp**。

R4: **74.6224 ± 0.0457%**；相对 B 的配对增益为 **+1.7857 ± 0.1571 pp**。

## 2. R4 Teacher Agreement 与错误归因

以下是 Teacher 伪标签准确率，不是最终 Student 准确率。所有真实目标标签仅用于训练完成后的离线分析。
投票一致定义为：原始 Base+Source SPL 的预测，与当时已准入且不属于当前目标域的每个 Target Prompt 预测相同。
它不是 Source Prompt 单独预测的三教师多数投票，训练仍在相似度层面融合。

| 最终 Teacher 指标（三 seed） | 所有任务 | Clipart 作为目标域 |
|---|---:|---:|
| 原始 SPL Accuracy | 72.725% | 52.733% |
| R4 Cross-target Teacher Accuracy | 74.688% | 54.626% |
| 原始错误被纠正 / 全部样本 | 4.022% | 4.745% |
| 原始正确变错 / 全部样本 | 2.059% | 2.851% |
| 双方相同错误 / 全部样本 | 18.735% | 32.908% |
| 投票一致且错误 / 全部样本 | 12.946% | 21.232% |
| 投票与 Cross Teacher 同为相同错误 / 全部样本 | 12.712% | 20.837% |
| 投票共同错误但 Cross 融合纠正 / 全部样本 | 0.125% | 0.153% |
| 置信度 ≥0.7 且投票一致错误 / 全部样本 | 3.565% | 3.719% |
| 投票一致样本的 Precision | 83.142% | 68.697% |
| 原始 SPL 与 Cross Teacher 一致率 | 89.402% | 82.790% |

| 准入节点 | Cross Teacher 准入前 | 准入后 | 即时变化（pp） |
|---|---:|---:|---:|
| 250 | 73.733% | 74.369% | +0.636 |
| 500 | 74.263% | 74.587% | +0.324 |
| 750 | 74.570% | 74.650% | +0.080 |

实际参考 Prompt 的一致率另外计算：Source Prompt + 两个其他 Target Prompt，排除固定 Base CLIP。

| 最终实际 Prompt 指标 | 所有任务 | Clipart 作为目标域 |
|---|---:|---:|
| 三 Prompt 一致率 | 70.554% | 55.255% |
| 三 Prompt 一致且错误 / 全样本 | 9.229% | 14.682% |
| 三 Prompt 一致且 Cross 同样错误 / 全样本 | 8.925% | 14.110% |
| 三 Prompt 一致样本 Precision | 85.890% | 73.454% |

准入前后使用同一个 Student 与 Source 状态，因此上表衡量教师加入的即时伪标签效果，不是 Student 训练收益。
第一次准入作用最大，后续收益递减。Clipart 在所有 Source 下均排最后：自身准入不改变其 SPL，验证了自身教师排除规则。
Clipart 被纠正的样本中，约 91%–93% 至少有一个跨域 Teacher 的首选类别正确；这属于支持证据，不能直接视为因果贡献。
Clipart 的 Computer、Sink、Marker、Mop 等类别仍有明显共同错误，单纯提高投票一致率无法处理这些错误。

### 固定教师的 Leave-one-out 贡献

加载原 R4 Source 最终 Checkpoint 的完整 Prompt 参数进行 CPU 推理，不执行任何训练更新。移除一个 Target Teacher 后重新融合其余相似度；不重训 Student。
下表为 Teacher 伪标签的边际准确率变化，正值表示加入该 Teacher 有益。

| Source | 当前 Target | 被移除 Teacher | 加入该教师的净贡献（pp） |
|---|---|---|---:|
| Art | Clipart | Product | +0.244 |
| Art | Clipart | Real World | +0.145 |
| Art | Product | Clipart | +0.533 |
| Art | Product | Real World | +0.743 |
| Art | Real World | Clipart | +0.092 |
| Art | Real World | Product | -0.015 |
| Clipart | Art | Product | +0.481 |
| Clipart | Art | Real World | +0.233 |
| Clipart | Product | Art | +0.503 |
| Clipart | Product | Real World | +0.398 |
| Clipart | Real World | Art | +0.138 |
| Clipart | Real World | Product | +0.574 |
| Product | Art | Clipart | +0.659 |
| Product | Art | Real World | +0.700 |
| Product | Clipart | Art | -0.122 |
| Product | Clipart | Real World | +0.092 |
| Product | Real World | Art | +0.184 |
| Product | Real World | Clipart | +0.115 |
| Real World | Art | Clipart | -0.233 |
| Real World | Art | Product | +0.206 |
| Real World | Clipart | Art | +0.092 |
| Real World | Clipart | Product | +0.466 |
| Real World | Product | Art | -0.263 |
| Real World | Product | Clipart | -0.405 |

CPU 重算与已保存 GPU Cross Teacher 预测共有 1 个样本点差异；错误分区始终使用原始保存预测。

## 3. 固定 R4 教师的 β 扫描

以下第一轮使用 seed 42。`FUniform` 复用原 R4 的三个教师快照，按原 250/500/750 时刻开放，必须逐项复现 R4 参数。
β 扫描共享同样的 Snapshot、Bank、样本流、更新预算与准入时间，只改变距离 Softmax 的 β。
β=0 仍保留 Invalid Centroid Mask；因此与普通 Uniform 的差异可包含缺失类别 Mask 的影响。

| 方法 | 12-task Macro | 相对 R4（pp） | 最终多有效域 One-hot 比例 | Clipart 的 Source 权重 |
|---|---:|---:|---:|---:|
| R4 | 74.6470% | +0.0000 | — | — |
| R5 | 74.2683% | -0.3788 | — | — |
| FUniform | 74.6470% | +0.0000 | 0.00% | 33.33% |
| FBeta0 | 74.6224% | -0.0247 | 0.00% | 33.33% |
| FBeta0p1 | 74.6557% | +0.0086 | 0.00% | 33.77% |
| FBeta1 | 74.6884% | +0.0413 | 0.00% | 37.42% |
| FBeta10 | 74.3588% | -0.2882 | 28.07% | 45.59% |

One-hot 使用有至少两个有效质心的样本×类别位置，阈值为最大权重 >0.99；排除只有 Source 可用的平凡单教师位置。
Source 权重是在固定 Base 系数 0.5 之外的参考教师半边内计算。实际训练批次统计另见 `actual_teacher_contributions.csv`。
原 R5 的后续快照受其自身训练影响，不能将 R5 与新 β 扫描的差异全部归因于温度。

### 几何距离与 Teacher 能力的关系

`geometry_mismatch.csv` 使用目标真标签做离线诊断，绝不反馈给 Bank。考察真实类别位置的教师权重，
并统计这些权重分给首选类别正确教师的总量。该总量不是经过校准的正确概率，也不等于融合后的准确率。

## 4. Progressive Admission 对照

| 方法 | Seed 42 Macro | 相对 R4（pp） | Teacher 来源与时机 |
|---|---:|---:|---|
| R4 | 74.6470% | +0.0000 | 在线、逐步生成并准入 |
| FUniform | 74.6470% | +0.0000 | 同样 R4 快照、原时刻准入 |
| FStartInit | 72.5806% | -2.0664 | Step 0 全部初始未训练 Prompt 快照，随后固定 |
| FStartRef | 74.6533% | +0.0063 | Step 0 全部已训练 R4 快照，离线时机对照 |

`FStartInit` 同时改变教师成熟度与准入时机；它单独不能证明渐进排序必要。
`FStartRef` 与 `FUniform` 教师质量和身份完全相同，因此能隔离开放时间；但使用未来教师快照，不能作为因果在线算法。
所有 Student 的 Source 与每个 Target 各 1,000 次更新。参考快照的共同 R4 准备另需每分支 1,000 次更新，
由这些离线对照共享；与原 R4 比较时不能声称端到端计算预算相同。

### Seed 42 的 12 个任务准确率

| Source → Target | R4 | FBeta0 | FBeta0p1 | FBeta1 | FBeta10 | FStartInit | FStartRef |
|---|---:|---:|---:|---:|---:|---:|---:|
| Art → Clipart | 54.525 | 54.525 | 54.616 | 54.639 | 53.998 | 50.584 | 54.822 |
| Art → Product | 83.893 | 83.532 | 83.622 | 83.645 | 83.735 | 81.032 | 83.848 |
| Art → Real World | 85.104 | 85.127 | 85.127 | 85.242 | 84.921 | 84.531 | 85.196 |
| Clipart → Art | 74.413 | 74.413 | 74.413 | 74.454 | 73.218 | 72.023 | 74.372 |
| Clipart → Product | 84.118 | 84.118 | 84.253 | 84.388 | 83.645 | 81.280 | 84.141 |
| Clipart → Real World | 85.219 | 85.219 | 85.265 | 85.104 | 84.393 | 83.452 | 85.334 |
| Product → Art | 73.548 | 73.506 | 73.465 | 73.383 | 73.589 | 72.105 | 73.589 |
| Product → Clipart | 53.929 | 53.929 | 54.021 | 54.204 | 53.631 | 52.325 | 54.044 |
| Product → Real World | 85.127 | 85.127 | 85.219 | 85.127 | 85.242 | 84.967 | 84.967 |
| Real World → Art | 74.825 | 74.907 | 74.825 | 74.742 | 74.372 | 72.600 | 74.578 |
| Real World → Clipart | 55.166 | 55.166 | 55.143 | 54.960 | 55.349 | 52.852 | 55.143 |
| Real World → Product | 85.898 | 85.898 | 85.898 | 86.371 | 86.213 | 83.217 | 85.808 |

## 5. 执行状态、证据与限制

保留原 A/B/C 和 R0–R6。新增入口为 `train_b_followup.py`，运行树为 `runs/b_followup/`。
完整任务准确率见 `accuracy_tasks.csv`，Source 表见 `source_summary.csv`，配对 seed 统计见 `paired_comparisons.csv`。
错误分区、类别分区和教师边际贡献见 `agreement/`；预算、参考快照一致性与原 R4 精确复现见 `validation.json`。
不新增训练损失、过滤、阈值调节或特征变换；目标标签不参与训练、准入、Bank 或 Checkpoint 选择。

预注册追加 seed 条件：相对 R4 ≥0.5 pp 且至少三个 Source 正收益。第一轮候选如下：

- FBeta0: -0.0247 pp，1/4 Source 正收益，追加 seed 条件不满足。
- FBeta0p1: +0.0086 pp，2/4 Source 正收益，追加 seed 条件不满足。
- FBeta1: +0.0413 pp，4/4 Source 正收益，追加 seed 条件不满足。
- FBeta10: -0.2882 pp，1/4 Source 正收益，追加 seed 条件不满足。
- FStartRef: +0.0063 pp，2/4 Source 正收益，追加 seed 条件不满足。
- FStartInit: -2.0664 pp，0/4 Source 正收益，追加 seed 条件不满足。

## 6. Clipart 的类别条件几何诊断

| β | Source 在真实类别的权重 | Source 在其他类别的平均权重 | 真实类别的正确教师投票权重总量 |
|---|---:|---:|---:|
| 0 | 33.333% | 33.333% | 52.201% |
| 0.1 | 33.641% | 33.767% | 52.244% |
| 1 | 36.166% | 37.436% | 52.616% |
| 10 | 42.145% | 45.641% | 53.607% |

这里的教师集合、Bank 与 Source 最终 Prompt 固定，因此仅度量距离权重变化。
真实类别的正确投票权重增加，并不保证整个 65 类融合分类器改进；其他类别的权重和教师相似度也影响决策。
扫描 β 同时改变集中程度及教师偏好，不能把全部变化唯一归因于 One-hot 本身。

已完成并审核 28 个新增 run；FUniform 全 Prompt 与 R4 一致，扫描 Bank 逐项一致，Source 轨迹和预算保持一致。
