# 统一预算 VLM MTDA 比较：结论与建议

本轮16次训练全部完成。统一 seed=42、RN50、每域1,000次更新、batch=30，一个模型同时适配三个目标域；完整目标域评估，固定最终检查点。

## 是否只是 CLIP 带来的好成绩？

CLIP Zero-shot 已达到72.14%，说明绝对准确率很大程度依赖CLIP的预训练能力。但R4在相同CLIP、相同B分类器结构和相同训练预算下，相比B/R0M增加1.6483 pp；既有三种子的配对提升为1.7857 ± 0.1571 pp。这个受控差值支持跨目标域Teacher带来额外收益。

DAPrompt论文阈值版本73.33%，代码阈值版本73.26%；AD-CLIP代码行为72.75%，投影层登记修复71.70%。R4为74.65%，在本轮配置下高于全部四种预先确定的比较配置。

| 比较对象 | R4整体增益(pp) | R4获胜Source数 | R4在Clipart目标的增益(pp) |
|---|---:|---:|---:|
| DAPrompt | 1.32 | 4/4 | 0.84 |
| DAPrompt-CodeTau | 1.39 | 4/4 | 1.09 |
| ADCLIP-Code | 1.90 | 4/4 | 1.37 |
| ADCLIP-Registered | 2.94 | 4/4 | 2.89 |

## 改造范围与解释边界

DAPrompt延展为一个共享generic context和四组domain vectors，按已知域身份选择目标域自身分类器。三个目标域损失取均值，保留原生硬伪标签与接受样本CE。AD-CLIP使用一个共享的图像条件Prompt投影网络，三个Source–Target配对损失取均值，保留原生CE、MSE、KL和entropy；没有将这些损失加入我们的SPL入口。

AD-CLIP原代码的部分Linear层放在普通Python list中，不进入优化器和检查点。代码行为版本保持其固定随机投影；修复版本登记并训练这些层。两者都明确保留原代码的Source输入mapping loss，与论文Target输入不同；目标域单图推理按论文描述补齐。原代码还重复同一个content decoder，并对batch维度L2归一化后计算entropy。这些限制已记入[协议](../../docs/vlm_competitors_protocol.md)。登记修复的负收益只说明本轮原生学习率和短预算下并未改善，不能据此证明原论文方法能力较弱。

排名差异还包含不同优化器、损失、共享方式及模型容量。DAPrompt可训练参数40,960；AD-CLIP代码行为1,314,821，登记修复5,743,109。它们均使用30,000次Source样本抽取；AD-CLIP同一Source batch与三域配对，对应90,000个条件Source CE视图，不能称为完全相同的计算预算。

## 计算开销与验证

每个Source设置的平均时间（包含初始/最终评估，已复用冻结视觉缓存）：
- DAPrompt: 3.38分钟。
- DAPrompt-CodeTau: 3.38分钟。
- ADCLIP-Code: 48.98分钟。
- ADCLIP-Registered: 49.55分钟。

57项pytest测试通过；16份训练历史均为有限值并有完整1,000步；每域样本抽取次数30,000；所有目标域尾批都纳入评估。16个检查点均加载到不同初始化的新模型，全部保存的张量逐项相等，每目标域96个分散样本的预测无差异。原A/B/C、可靠性实验及后续实验的代码哈希保留。

## 下一步建议

保留R4作为主方法，并用本轮的受控B/R0M对照作为跨Teacher收益的主要因果证据。若准备论文级比较，下一步应先补DAPrompt多seed，随后单独定义并验证AD-CLIP的论文一致实现（尤其mapping训练输入和content decoder）；本轮不追加这些算法改动。原生长预算和随机增强应作为另一组明确分开的协议，不应将本轮1,000步结果直接称为论文复现。

详细四Source和12任务结果见[完整报告](report.md)、[结果表](accuracy_12_tasks.csv)及[验证记录](validation.json)。
