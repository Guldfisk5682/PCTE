# 研究结论与下一步建议

下列全变体结果是固定 seed42、四个 Source、12 个迁移任务的最终 checkpoint。有收益的 Uniform 方案及其对照另补 seeds41/43；多 seed 结果写在独立目录，不将其他变体伪装成三 seed。

## 四个问题

| 问题 | 受控比较 | Macro 差值（百分点） | 首轮结论 |
|---|---|---:|---|
| E2H 顺序本身 | R1 − R0M | +0.22 | 提升小；不能解释为跨 Target 参数传递 |
| 跨目标 Prompt Teacher | R2 − R1 / R4 − R0M | +1.29 / +1.65 | 两种组织均改善，且四个 Source 都改善 |
| 类条件 Bank 距离权重 | R3 − R2 / R5 − R4 | -0.35 / -0.38 | 当前 beta=10 未优于 Uniform |
| 周期刷新 | R6 − R5 | -0.16 | 当前 interval200 未改善最终 Macro |

所有方法每个 Target 各 1,000 次更新，Source 共 1,000 次更新；R0M 的所有 Prompt 与原始 B 在真实 1,000-step 运行中逐元素相同。顺序训练每三步更新 Source，因此阶段结束时 Source 更新数为 334/667/1,000；R1 的小幅变化仍包括 Source Teacher 成熟度时序。B 的 Target 参数独立，不能将它称为直接的跨 Target 课程迁移。

## Clipart 目标域

| R0 | R1 | R2 | R3 | R4 | R5 | R6 |
|---:|---:|---:|---:|---:|---:|---:|
| 52.73 | 52.74 | 54.61 | 54.19 | 54.54 | 54.20 | 54.21 |

这行是 Clipart **作为 Target** 时对三个 Source 的宏平均；Source=Clipart 的三目标平均在主报告另列。Uniform 增益在 Clipart 上也可见。仍应把约 54%–55% 的结果视为未解决的难点。

## 为什么好 Bank 仍不带来好权重

| Bank 变体 | 平均类别覆盖率 | 接受样本精度 | Oracle cosine |
|---|---:|---:|---:|
| R3 | 99.62% | 92.45% | 0.9759 |
| R5 | 99.49% | 92.75% | 0.9759 |

覆盖率与 Oracle cosine 高，说明第一版的高置信度 Bank 能形成接近真实类中心的质心；但它们是类中心质量指标，不是跨域 Text Prompt 的可靠度指标。在 R3 Art→Clipart 的实际训练中，Source 平均参考权重约 82%；Uniform 三 Teacher 的 Source 权重为 1/3。Base 始终占 1/2，因此 Source 在整条 SPL 融合中的系数约从 1/6 增到 0.41。这与 Source Teacher 重新主导有关；Raw-L2 邻近是否主要反映风格或特征幅值，还需要后续受控消融验证。

beta=10 并非所有样本都绝对 one-hot。例如 R3 Art→Clipart 的最终全缓存诊断，在多个有效 Teacher 的类别×样本条件下约 44.45% 权重超过 0.99，权重熵约 0.245。应同时看 Source 占比、有效 Teacher 数和条件权重熵，不能把只有一个有效 Teacher 的正常权重一混作饱和。

## 刷新结果

R6 总计 48 次 Bank 建立/重建，其中 36 次周期刷新。接受样本 precision 变化 15 次为正、21 次为负，平均变化为 -0.0065 个百分点。这不是稳定的质量提升，且最终准确率也没有改善。

R6 同步更新 Prompt Teacher Snapshot 和 Bank；它检验的是两者作为同版本的动态刷新，不能把任何差异单独归因于 centroid 的刷新。最后一次 step1000 刷新只贡献质量诊断，不会影响之后的学习。按预设收益条件，本轮不展开 interval100/400。

## 后续工作建议

1. **保留 R4 作为主要候选增强路径。** Joint Uniform 避免三阶段组织，首轮表现与 R2 相当；以 R2/R1、R4/R0M 的配对三 seed 差值决定稳定性。R0 始终保留为正式 B 基线。
2. **先做 beta 的受控敏感性，再修改 Bank 机制。** 保持所有 Snapshot、阈值0.7、准入节点和 Loss 不变，比较 beta=0/0.1/1/10。当前覆盖率接近完整，beta=0 是接近 Uniform 的可解释边界；重点看 Clipart 的 Source 权重与伪标签改错/引错。该实验尚未运行。
3. **把 Source Teacher 时间效应与 E2H 分开。** 可用已有手动顺序接口比较 E2H/H2E；若要判断纯排序，需另设同一 Source Teacher 时序的明确预算对照，而不修改 R0。
4. **刷新前先核对 Teacher 是否更可靠。** 若以后继续刷新研究，添加 Uniform Teacher Snapshot Refresh 对照，与固定 Uniform 比较，再判断距离 Bank 是否提供额外价值。当前结果不足以优先扩大刷新扫描。

这些建议均围绕已有 Prompt Teacher、Bank 距离、排序和刷新；未引入或验证额外算法模块。

## 与已有文献调研的关系

[Luna 文献备忘录](../../docs/spl_reliability_literature.md)说明 CRPL 的权重在参考域轴上归一化，单源时退化为一。本轮扩展 Target bank 后恢复了域间竞争，但实验显示“视觉类中心接近”不能直接当作“Text Teacher 跨域预测可靠”。SHOT 等目标原型自举工作也有不同的可训练表示和损失设定，因此这里只借鉴质心与伪标签的诊断思路，没有移植其额外优化模块。

完整 12-task 表、Source/Target macro、Loss/Accuracy 曲线、Teacher 贡献、刷新统计和离线混淆矩阵见同目录 CSV；所有正式 accuracy 使用最终固定预算 checkpoint。

## 配对三 seed 确认

| 方法 | 总体 Macro（mean ± sample std） |
|---|---:|
| R0 | 72.84 ± 0.15 |
| R0M | 72.84 ± 0.15 |
| R1 | 73.26 ± 0.05 |
| R2 | 74.47 ± 0.03 |
| R4 | 74.62 ± 0.05 |

| 配对比较 | 差值 mean ± sample std（百分点） |
|---|---:|
| R1 − R0M | +0.42 ± 0.18 |
| R2 − R1 | +1.21 ± 0.08 |
| R2 − R0 | +1.63 ± 0.13 |
| R4 − R0M | +1.79 ± 0.16 |
| R4 − R2 | +0.16 ± 0.04 |

三个 seed 确认了 Uniform 的主要收益：E2H Uniform 相对 R1 的配对增益约 +1.21 ± 0.08；Joint Uniform 相对 B 的配对增益约 +1.79 ± 0.16。R4 仅比 R2 高约 0.16 点，两者均应保留作对照，但 Joint 路径更便于后续控制 Teacher 准入和刷新。


[三 seed 配对明细](multiseed/report.md)保留每个任务与 Source 的差值，其余 Bank/Refresh 变体仍为首轮 seed42；没有填补未运行的 seeds。
