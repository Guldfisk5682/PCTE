# SPL 伪标签可靠性文献备忘录

**范围。** 本备忘录只整理文献、论文公式和作者公开实现，不实施或验证新算法，也不使用 Office-Home 目标真标签。当前实验设定和 seed 42 基线（共享 MTDA 72.74%、独立 class MTDA 73.00%、pairwise UDA 72.65%、zero-shot 72.14%）来自本任务上下文，未在本轮复核。当前 teacher 为 `softmax(CLIP_scale * (0.5 z·F_base + 0.5 z·F_source))`，训练目标仅为 `source CE + 0.5 target soft CE`；CLIP RN50 冻结，只训练 class/domain prompts。以下任何方向都只是后续独立消融候选，不是效果承诺。

## CRPL 的 source centroid bank 在单源时发生什么

CRPL（CVPR 2025）为每个源域、每个类别维护一个类质心。论文定义未归一化图像特征的源类质心 `c^i_k = mean(z_pre | source=i, y=k)`，对固定类别 `k`，以目标样本 `z_pre` 到各源域该类质心的距离产生权重，再把权重用于源域 prompt 特征，和 base prompt 特征共同生成 teacher 分布。直观上，多源时它回答的是“对这个类别，这个目标样本更应参考哪个源域”。它不是目标类别分布估计器，也不是直接比较不同类别的源类质心。

在单源情况下，源域轴只有 `i=1`。作者公开代码创建的 `running_means` 形状为 `[n_domains, n_cls, feature_dim]`，并计算 `weights = softmax(dim=0)(-distance * w_scale)`；距离张量形状为 `[n_domains, batch, n_cls]`。对任意有限 softmax 输入 `-w_scale d_1`，包括把零初始化的未见类质心代入距离函数所得的有限值，单元素 softmax 都满足 `softmax([-w_scale d_1]) = 1`。这是源轴权重的数学性质；但作者代码在 `running_count` 显示所有源域类别都被观测到之前，会跳过距离权重分支。因此，实际代码不会在未见类仍为零初始化时把它送入该分支。源轴权重不携带距离排序信息；进入该分支时，它把唯一源 prompt 以固定系数加入 teacher。CRPL 论文中的目标 soft label 仍会依赖类间文本相似度，但**源轴权重本身**不能在类别之间挑选更可信的源类原型。

这不代表源类质心在单源场景中没有信息。质心集合 `{c_s,k}_{k=1..K}` 仍可沿**类别轴**比较目标特征与各类别原型；也可以结合每个目标域的无标签邻域结构，判断 teacher 的类别分配是否与视觉空间相容。这是对已有几何信息的另一种使用方式，属于待独立检验的研究假设，不是 CRPL 原样方法。若只把 CRPL 的源域 softmax 维度保留为长度 1，它不能实现这种类别匹配。

一个复现细节：本轮无法取得 CRPL 原 PDF 的可判读页图，因此不判断论文公式正负号，也不引用无法从页图核对的节号/公式号；不保留此前基于可检索文本的“公式/代码符号不一致”断言。已直接核对作者公开 `main.py`：权重实现为 `softmax(-distance * w_scale, dim=0)`。质心建在未归一化视觉特征上，而余弦分类用归一化图像/文本特征；若比较两类几何量，需先固定并说明度量空间，避免把 raw-L2 距离和 cosine 分数不加校准地混合。

Office-Home 中 Clipart 是 CRPL 讨论的困难目标域之一：论文表格报告 zero-shot CLIP 在 Clipart 上为 50.4%，CRPL 多源设置为 63.5%，其 source-combined 设置为 60.4%。这些数值来自该论文协议，不能和当前单源、多目标、seed 42 结果直接比较。CRPL 的标准 UDA 问题是若干带标签源域适配到一个无标签目标域；本任务把一个源域信息用于多个目标域，方向和采样结构不同。

## 直接相关论文

| 论文与年份/会议 | 几何或伪标签机制 | 训练时目标真标签 / 目标数据 | 与当前约束的关系 |
|---|---|---|---|
| [Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation (CRPL)](https://openaccess.thecvf.com/content/CVPR2025/html/Vuong_Preserving_Clusters_in_Prompt_Learning_for_Unsupervised_Domain_Adaptation_CVPR_2025_paper.html), CVPR 2025；[作者代码](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning) | 源侧用 `c^i_k` 距离对每类的源 prompt 参考加权；目标侧用 source-enhanced soft pseudo-label 指导 target prompt。论文还描述基于 Wasserstein/OT 的目标簇匹配。 | 不用目标真标签训练；用目标图像及模型生成的软伪标签。 | 论文与代码均显式使用源类质心。标准设置有多源/一个目标；多源轴对单源退化。OT 是当前损失契约之外的独立组件。论文报告均衡簇先验；对不均衡类先验的影响未经当前协议验证。 |
| [Domain Adaptation via Prompt Learning (DAPL)](https://doi.org/10.1109/TNNLS.2023.3327962), IEEE TNNLS（2023 年 online；36(1) 卷期为 2025） | 将域特定 prompt 与域共享/类别 prompt 分开；source CE 与 target 伪标签 CE 学习不同域的条件分类器。目标 pseudo-label 由 CLIP zero-shot 产生，并按置信阈值筛选（Office-Home 报告阈值 0.6）。 | 目标真标签不用作训练信号；需要无标签目标图像和 zero-shot 伪标签。 | 与当前可比较的差异：文中使用阈值后的目标 pseudo-label；当前方案使用软 teacher 且不筛选。该机制对当前设置的适用性未验证。 |
| [PADCLIP: Pseudo-labeling with Adaptive Debiasing in CLIP for UDA](https://openaccess.thecvf.com/content/ICCV2023/html/Lai_PADCLIP_Pseudo-labeling_with_Adaptive_Debiasing_in_CLIP_for_Unsupervised_Domain_ICCV_2023_paper.html), ICCV 2023 | 从 zero-shot 预测生成伪标签，以因果 debias、动态 debias factor/momentum 和 catastrophic-forgetting measurement 调整目标学习。 | 不用目标真标签训练；用目标图像和模型伪标签。 | 与当前差异：论文微调 CLIP 视觉分支，而当前冻结 RN50、仅 prompt 可训；该机制对当前约束的适用性未验证。 |
| [Multi-Prompt Alignment for Multi-Source UDA (MPA)](https://proceedings.neurips.cc/paper_files/paper/2023/file/eadeef7c51ad86989cc3b311cb49ec89-Paper-Conference.pdf), NeurIPS 2023 | 针对不同源—目标对学习 prompt，再用去噪自编码器把各对 prompt 投影/对齐到共享子空间，鼓励重建 prompt 一致。 | 目标真标签不用作训练；目标图像用 CLIP zero-shot 置信阈值后的静态硬伪标签。 | 论文设置是多源到单目标；虽然其 CLIP encoder 冻结且只学习 prompts/auto-encoder，当前单源多目标下的对齐策略适用性未验证。 |
| [Enhancing Domain Adaptation through Prompt Gradient Alignment (PGA)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/50ea4cf434c84a50203d7145afc7b853-Abstract-Conference.html), NeurIPS 2024 | 将每个域损失视为一个优化目标，对齐 prompt 梯度方向并惩罚梯度范数；论文报告单源与多源 UDA 变体。 | 目标真标签不用作训练；依赖目标图像及目标侧学习信号。 | 与当前可比较的差异是目标为 per-domain prompt 梯度冲突；它不是类质心或邻域可靠度方法。其适用性未在当前多目标设置验证。 |
| [Do We Really Need to Access the Source Data? SHOT](https://proceedings.mlr.press/v119/liang20a.html), ICML 2020 | 在目标特征上先以当前 softmax 输出作权重计算 `c_k^(0)=Σ_x p_k(x)g(x)/Σ_x p_k(x)`；再按到 `c_k^(0)` 的 cosine 距离硬分配 pseudo-label，按硬标签重算 `c_k^(1)` 并重新分配，可迭代多轮。论文称实验中迭代一次已足够。 | 目标真标签不用作训练；需要无标签目标图像集合。 | SHOT 是目标域自举质心（初始 soft、随后 hard），不是源类质心法；其 source-free 设定还会更新特征提取器，和当前冻结视觉 backbone 不同。 |
| [Exploiting the Intrinsic Neighborhood Structure for Source-free DA (NRC)](https://proceedings.neurips.cc/paper/2021/hash/f5deaeeae1538fb6c45901d524ee2f98-Abstract.html), NeurIPS 2021 | 用目标样本的局部、互为近邻及扩展邻域亲和度，鼓励高亲和样本预测一致，并用 reciprocal neighbors 降低错误邻居影响。 | 目标真标签不用作训练；需要无标签目标数据集合以建立邻域。 | 与当前可比较的差异是 NRC 用 source-free 表征适配；固定 CLIP 特征上的轻量 soft-label 校准并非其验证设置。 |
| [Transferrable Prototypical Networks for UDA (TPN)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Transferrable_Prototypical_Networks_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf), CVPR 2019 | 用带标签源类原型为目标样本分配最近原型伪标签，再计算源、目标及混合原型；以原型距离和类别得分分布 KL 约束适配。 | 不用目标真标签训练；用源标签、无标签目标图像和伪标签。 | 与当前可比较的差异是 TPN 沿类别轴用源原型指派伪标签，但端到端更新 embedding；当前设置下此机制尚未验证。 |
| [Source-free Domain Adaptation with Class Prototype Discovery (CPD)](https://www.sciencedirect.com/science/article/abs/pii/S0031320323006726), Pattern Recognition 145 (2024), article 109974；[DOI](https://doi.org/10.1016/j.patcog.2023.109974) | 通过目标分类分数优化语义类原型，用原型聚类目标特征生成伪标签，并加入目标分布/原型正则。 | 目标真标签不用作训练；用无标签目标图像，且不需要源数据参与适配阶段。 | 与当前可比较的差异是 CPD 在 source-free 设置下估计目标类原型；源自由设置和可更新表征与当前不同。迁移适用性未验证。 |
| [Spherical Space DA with Robust Pseudo-Label Loss (RSDA)](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Spherical_Space_Domain_Adaptation_With_Robust_Pseudo-Label_Loss_CVPR_2020_paper.html), CVPR 2020 | 在 L2 球面特征空间中，用伪类别的类中心距离及 Gaussian-uniform 混合模型估计标签正确概率，再软加权伪标签损失。 | 目标真标签不用作训练；使用无标签目标图像和伪类别。 | 提供“距离用于连续可靠度权重而不是硬筛选”的先例。它学习视觉表示、用对抗适配，不是冻结 RN50 + prompts 的直接可移植实现。 |
| [Domain Prompt Learning for Efficiently Adapting CLIP to Unseen Domains](https://www.jstage.jst.go.jp/article/tjsai/38/6/38_38-6_B-MC2/_article), Transactions of the Japanese Society for Artificial Intelligence 38(6), 2023；[arXiv 预印本](https://arxiv.org/abs/2111.12853) | 从输入图像推断域表示，再由轻量 prompt generator 生成适配 CLIP 的域 prompt；训练仅调 prompt generator。 | 没有目标域训练图像或目标标签，也没有目标训练伪标签；属于 domain generalization。 | 该论文没有 transductive 目标数据；其条件 prompt 机制与当前任务的关系仅作为背景记录，目标数据适配适用性未验证。 |

## 证据索引

- **CRPL 质心与权重：** 已核对指定仓库版本的官方源文件：本地 `HEAD` 为 `2c032a62fc706b643bfc30f49c9a064dd24aa7ac`。固定版本 [`main.py` 初始化 centroid bank，第 251–254 行](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/2c032a62fc706b643bfc30f49c9a064dd24aa7ac/main.py#L251-L254)、[累积域/类质心，第 309–317 行](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/2c032a62fc706b643bfc30f49c9a064dd24aa7ac/main.py#L309-L317)、[检查类别是否均已观测、计算距离并沿源域轴 softmax，第 350–356 行](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/2c032a62fc706b643bfc30f49c9a064dd24aa7ac/main.py#L350-L356)、[生成 detached soft target，第 365–367 行](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/2c032a62fc706b643bfc30f49c9a064dd24aa7ac/main.py#L365-L367)。[CLIP 参数冻结，第 156–157 行](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/2c032a62fc706b643bfc30f49c9a064dd24aa7ac/main.py#L156-L157)；[优化器仅接收 prompt learner 参数，第 228–238 行](https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning/blob/2c032a62fc706b643bfc30f49c9a064dd24aa7ac/main.py#L228-L238)。CVF 官方论文页可查论文正文和结果；原 PDF 未取得可判读页图，所以这里不报论文节号、公式号或页码，也不判断 Eq. 8 正负号。代码行号均用本地固定版本逐项核对，不沿用网页抽取行号。
- **DAPL：** arXiv/期刊一手页面提供方法与实验说明；本轮未用原 PDF 页图核对节号、公式号或页码，故不引用这些编号。Office-Home 置信阈值来自论文实验设置文字。题名/摘要的 TNNLS 书目由 DOI `10.1109/TNNLS.2023.3327962` 对应，2023 是 online 年，卷期为 2025。
- **PADCLIP：** ICCV 2023 CVF 官方论文页摘要用于核对 zero-shot 伪标签及 adaptive debias 的高层机制；本轮未核对原 PDF 页图，因此不引用图号、节号或公式编号。
- **MPA：** 已直接核读 NeurIPS 2023 官方 PDF：§3.2 Eq. (3)–(4) 给出 pairwise prompt 与 static thresholded pseudo-label 目标；同节 Eq. (6)–(9) 定义 auto-encoder prompt denoising/alignment，§3.3 Eq. (10) 是 latent subspace tuning；PDF 第 3 页说明训练时 CLIP image/text encoder 均冻结。
- **PGA：** 已直接核读 NeurIPS 2024 官方 PDF §4.3 中 single-source source/target prompt gradient alignment 的描述，以及摘要、§4 和 Appendix B 中 single-source 与 multi-source 版本的描述。
- **SHOT：** 已直接核读 ICML 2020 PMLR PDF §3.3，Eq. (4) 用当前预测概率初始化目标类软质心 `c_k^(0)`，Eq. (5) 依 cosine 最近质心产生硬 pseudo-label，Eq. (6) 用硬标签重算质心并重分配，Eq. (7) 将 pseudo-label loss 与 information-maximization 目标共同用于更新目标 encoder；正文说多轮可做，但实验中更新一次已足够。相关位置在 PDF 第 3–4 页。
- **NRC：** 已直接核读 NeurIPS 2021 官方 PDF §3.1 Eq. (1)–(6) 中目标邻居预测一致性、reciprocal-neighbor affinity 与 self-regularization 的定义；expanded neighborhood 见后续 §3.2。论文明确为 source-free 目标特征适配。
- **TPN：** CVPR 2019 CVF 论文摘要直接说明 nearest source prototype 伪标签、source/target/mixed prototypes 距离匹配和 prototype score-distribution KL；本备忘录没有从可读页面抽取并核对具体公式编号。
- **CPD：** ScienceDirect 出版社论文记录的题名、摘要与 DOI `10.1016/j.patcog.2023.109974` 对应；摘要描述 target classification-score prototype optimization、prototype pseudo-label clustering 和 prototype regularization。
- **RSDA：** CVPR 2020 CVF 正式页面摘要及论文 PDF 描述球面空间伪标签权重、到类中心距离与 Gaussian-uniform mixture posterior；本轮仅据摘要记录机制，不推断它可直接迁移到 prompt-only 设定。
- **DPL：** J-STAGE 期刊原文记录确认 2023 年 TJS AI 38(6)，摘要说明 domain inference / conditional prompt generation、轻量 prompt generator，以及未见目标域的 DG 设定；arXiv 链接为 2021 预印本。

表格中的“关系”栏仅用于说明研究对象和训练协议的可比较差异，不构成对方法在当前协议中的适用性结论。候选方向也仅是待验证问题。

## 可供后续独立消融的方向（均为推论）

1. **源类原型沿类别轴形成第二教师。** 由带标签源训练集计算每类 `μ_s,k`，对每个目标域分别在冻结 RN50 的同一视觉空间中计算 `d_k(x)`，形成软原型分布 `r_k(x) ∝ exp(-d_k(x)/τ)`。然后单独比较：原 SPL teacher `q`；`r` 单独；以及保留软目标的融合 `q' = normalize((1-λ)q + λr)`。另一种更保守的版本是只用 teacher 与原型分布的一致程度给 target soft CE 乘连续样本权重，不硬过滤样本。关键假设是源类原型在目标域仍保持类间排序；对 Clipart 这种风格变化较大的目标，这个假设必须由无标签几何诊断和多 seed 结果检验，不能依据目标准确率挑参数。

2. **每个目标域内部的互近邻一致性。** 在冻结图像特征上对同一个目标域建立 kNN/reciprocal-kNN 图，用邻居的 detached soft teacher 分布做亲和度加权平均，观察 teacher 在局部流形上是否一致；可将一致度用于软校准或 target-loss 权重。不同目标域先分别建图，避免把目标域风格差异当成类邻近。该方案只用目标图像和伪标签，但会引入 transductive 依赖：目标集变化时邻接关系也变化；小 batch 内建图则不等同于 NRC 的全目标邻域。

3. **目标类簇到源类原型的类条件几何。** 以当前 soft teacher 做目标类加权均值 `μ_t,k = Σ_x q_k(x)z(x)/Σ_x q_k(x)`，考察它与 `μ_s,k` 的距离、类间 margin、以及随训练轮次的变化，再将这些几何量用于伪标签校准候选。不要先验要求每个目标类都贴近源原型；当 target style shift 大或 pseudo label 有偏时，自举质心会把错分扩大。可先作为无标签诊断量，随后再独立试软融合。

4. **目标分布平衡/OT 作为单独后续分支。** CRPL 的目标 prompt 与图像簇匹配直接触及 target cluster geometry，但其均衡簇先验以及 OT/hard assignment 会改变当前 `source CE + 0.5 soft target CE` 目标。若将来探索，应单独报告组件开关与类先验假设；不要把它解释为现有 SPL loss 的等价改写。

## 适用边界与证据限度

CRPL、DAPL、MPA、PGA 都围绕“有标签源 + 单个无标签目标域”的 transductive UDA 定义展开；文中多源指多个源域，而非一个源域同时供应多个目标域。NRC/SHOT/CPD 属于 source-free UDA，和当前仍保留源数据及 source CE 的协议不同。DPL 是 unseen-domain generalization，没有目标训练集。迁移到本项目时，几何方向比具体 loss 更能直接参考：源类中心给类语义锚点，目标邻域给目标流形一致性；二者各有失配和伪标签自举风险。

文献没有证明“固定 RN50、只学 class/domain prompts、使用当前两项损失”时，上述任一改动必然改善 Clipart 或三个 baseline。也没有用本项目 seed 42、多目标采样和 checkpoint 规则做过实验。本备忘录提出的候选均不需要目标真标签；后续实现若启动，仍须只用源标签训练 `source CE`，对目标训练数据只读取图像、生成 detached pseudo-label，并按仓库约束不以目标准确率选择 checkpoint。
