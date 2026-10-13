# CRPL → MTDA（SPL-only）迁移实验计划

> 交给 Codex 执行的工程任务书。目标是基于 **CRPL 官方实现**进行最小必要改造，在 Office-Home 上验证其 Prompt + Source Centroid Bank + Source-enhanced Pseudo-label（SPL）设计能否直接迁移到 **单源域、多目标域（MTDA）**。先验证原设计，不提出新的 MTDA 算法。

## 0. 任务边界与执行原则

**原始仓库**：https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning

**论文**：*Preserving Clusters in Prompt Learning for Unsupervised Domain Adaptation*（重点：§3.3、§5.1、§5.2；原论文 Eq. (4)–(9)、(12)，但本实验删除 OT）。

1. 首先 `git clone` 官方仓库；记录原仓库 URL、默认分支与原始 commit SHA，在独立开发分支上工作（建议 `feat/mtda-spl-baseline`）。**不要覆盖原仓库的 MSDA/CRPL 实验入口**。优先增加 MTDA 专用入口与模块，复用现有 CLIP、Prompt、数据读取逻辑。
2. 由用户另行告知 **Office-Home 数据集位置、CLIP 权重/缓存位置、可用设备**。本文不写具体路径。没有这些信息时，先完成代码实现与合成数据单测，不得伪称真实训练已完成。
3. 保留：冻结的 CLIP 图像/文本编码器、可学习 Shared/Class Prompt、可学习 Domain Prompt、单源域 Class Centroid Bank、Source CE 与 SPL Soft CE、目标域各自推理。
4. 删除本迁移实验中的：OT/Wasserstein、OT 伴随的最近原型 Hard CE、Entropy Regularization、伪标签置信度筛选、跨目标域对齐/蒸馏/一致性、新 Bank 加权规则，以及任何额外的 MTDA 专用网络模块。
5. 做 **A/B/C 三个变体**，训练协议、初始化、评估指标尽可能统一。不能通过读取目标训练图像的类别标签计算损失、生成伪标签、挑选模型或调参。
6. 不要只写代码。必须完成可运行的 smoke test、Office-Home 正式训练及结果汇总；有环境、数据或算力阻碍时，明确记录已完成与未完成的工作。

## 1. 任务定义与三个固定变体

设 Office-Home 有 4 个域。每次指定一个有标签源域 `S`，其余三个域构成无类别标签、但**域身份已知**的目标域集合 `T={T1,T2,T3}`。所有域共享同一套类别语义空间，Office-Home 为 65 类。这里的 MTDA 是 **1 source + multiple unlabeled targets**，不是原论文的 MSDA（multiple labeled sources + 1 target）。

| 变体 | 训练组织 | Class Context Token | Domain Context Token | Source Bank |
| --- | --- | --- | --- | --- |
| **A — Shared-MTDA** | `S+T1+T2+T3` 联合训练，一个模型 | **唯一** `P_cls`，源域与三个目标域完全共享、共同更新 | `P_S, P_T1, P_T2, P_T3`，各域独立 | 仅一个源域 Bank |
| **B — Independent-MTDA** | `S+T1+T2+T3` 联合训练，一个模型 | `P_cls,S` 与 `P_cls,T1/T2/T3` **各自独立**，不共享参数/梯度 | 同 A，各域独立 | 仅一个源域 Bank |
| **C — Pairwise-UDA** | 分别训练 `S→T1`、`S→T2`、`S→T3`，独立模型/优化器/随机状态 | **在每次单独训练内**，源与该目标域共享一套 `P_cls`；换一个目标域则重新初始化 | 每次训练各自拥有 `P_S`、`P_T` | 各个独立训练维护自己的单源域 Bank |

**B 的定义必须严格实现：**源域有自己的一套类别上下文，三个目标域各有自己的一套类别上下文；目标域类别上下文不直接接收 Source CE 梯度，只通过该域 SPL 更新。源域 Prompt 作为三个目标域共同使用的伪标签教师。不要擅自改成“每个目标域的类别 Prompt 也接收 Source CE”，那是另一个实验定义。

**C 的定义必须严格实现：**每个 `S→Ti` 都完整重置模型中所有可训练 Prompt、优化器、调度器和 Bank；但在这一对源/目标域之间，Class Prompt 共享。不能仅在一个已训练联合模型上切换目标域。

**解释边界：**B 的目标分支彼此不直接交换信息，只有共同源域教师，因此 B 与 C 的结果可能非常接近；这本身是需要记录的实验现象，不要为制造差异临时增加跨目标域损失。

## 2. Prompt 的形状、前向与初始化

记类别数 `K`（Office-Home = 65）、CLIP 文本 token embedding 宽度为 `d_ctx`、CLIP 视觉/文本对齐特征维度为 `D`。不要把 `d_ctx` 和 `D` 混淆。默认 `M1=M2=16`，保留 CLI 可配置。

对一个域 `q` 与类别 `k`：

`prompt(q,k) = [SOS] [P_cls(q)[k]] [P_domain[q]] [CLASS_k, punctuation, EOS/padding]`

- Class Context：`[K, M1, d_ctx]`。每类别独立；跨域是否共享由 A/B/C 决定。
- 每个 Domain Context：`[M2, d_ctx]`（若代码保留前置 1 维亦可）；同一域所有类别共享。
- 固定的 Class Name Token 与特殊 token 来自 CLIP 原始 tokenizer/token embedding，不训练。
- Text Features：`normalize(CLIP_text(prompt(q, all K classes)))`，形状 `[K, D]`。
- Image Features：`normalize(CLIP_image(x))` 作为分类相似度输入；另保留未归一化 `z_pre` 用于 Bank 统计。
- **冻结 CLIP 两个 Encoder 及 logit scale**；优化器参数组只能包含以上 Prompt 参数。

**初始化控制：**遵循官方 `model.py` 可学习 context 的 `normal_(std=0.02)`。为了消除 A/B/C 因随机初始化差异造成的混淆，针对同一源域与 seed：

1. 先生成一份**相同数值**的初始 Class Context 模板；A 直接共享该 Parameter；B 的源域及三个目标域分别 `clone` 为独立 Parameter；C 每个 `S→Ti` 使用同一份模板的独立拷贝。数值可以相同，**参数对象不得意外共享**（B/C）。
2. 源域/各目标域 Domain Context 也按 `(seed, domain_name, role)` 生成可复现初始化；对应域在不同变体的初始值应相同。
3. 完整保存 seed、初始化规则与所有实验配置。官方 `ctx_source_combined` 使用 `torch.empty` 而未显式初始化，**新实现不沿用这个多余且未初始化的分支**；一个源域只保留一个 Source Domain Prompt。
4. 不要重新加载或微调完整 CLIP；推理与训练的文本特征均通过当前对应域的 Prompt 生成。

梯度关系验证（自动化测试必须覆盖）：

- A：`P_cls` 同时收到 Source CE 与三个 Target SPL 的梯度。
- B：`P_cls,S` 仅由 Source CE 更新；`P_cls,Ti` 仅由其自身 Target SPL 更新；各 `P_Ti` 仅由该域 SPL 更新；`P_S` 仅由 Source CE 更新。
- C：当前 pair 的 `P_cls` 同时由 Source CE 与该目标域 SPL 更新，其他 pair 完全独立。

## 3. Source Centroid Bank（只保留一个源域）

保留 CRPL 的**源域视觉类别质心累计均值**，不建立目标域 Bank，不建立第二个源域/混合源域 Bank。

- `running_means`：`[1, K, D]`（内部使用 `[K,D]` 也行，但明确源域维度含义），`float32`，初始为 0。
- `running_count`：`[1, K]`，初始为 0。
- 每个源域训练 Batch 根据**真实源标签**与**冻结图像编码器输出的未归一化特征**更新：

$$
 c_{k}^{new}=\frac{n_k c_k^{old}+\sum_{j:y_j=k}z_j^{pre}}{n_k+m_k},\qquad n_k^{new}=n_k+m_k.
$$

- 只处理本 Batch 出现的类别；未见类别继续保留零计数。不能发生除零、NaN 或跨不同源实验的 Bank 污染。
- Bank 是统计状态，**不进入优化器/不反传梯度**，应随 checkpoint 保存以供检查与恢复。
- 记录每步或每个日志周期的 `observed_class_count/K`、累计样本数及关键断言；单测要与直接求平均的结果一致。

**单源域退化必须如实保留：**原 CRPL 的源域维度 softmax

$$
 w_{i,k}(x)=\operatorname{Softmax}_{i}\!\big(-s\lVert z^{pre}(x)-c_k^i\rVert_2^2\big)
$$

在 `N_source=1` 时，对所有样本和类别均为 `w_{1,k}(x)=1`。**当前实验仍维护 Bank，但它不改变 SPL 伪标签结果。**不要偷偷把 `distance` 改成跨类别 softmax、基于距离的置信度过滤、按目标域重加权或任何其他新规则。这些都留给以后单独设计消融。

## 4. SPL 与训练损失（本阶段唯一方法）

### 4.1 单源域 SPL 伪标签

对目标域 `Ti` 中样本 `x`：

1. 计算归一化视觉特征 `z`。
2. 使用冻结 CLIP **Base Text Prompt**（例如 `A photo of a [CLASS]`）得到 `F_base: [K,D]`。
3. 使用**当前已学习的源域 Prompt** 得到 `F_source: [K,D]`。A/C 使用当前 pair 共享的类别上下文和源域 Domain Prompt；B 使用源域独有的类别上下文和源域 Domain Prompt。
4. 因为仅一个源域，`w=1`，且没有独立的 source-combined 分支，按论文 Eq. (7)–(9) 的单源形式计算：

$$
 \ell_{teacher}(x,k)=\frac12(z\cdot F_{base,k})+\frac12(z\cdot F_{source,k}),
 \qquad
 \hat p(k\mid x)=\operatorname{Softmax}_k\big(s_{CLIP}\ell_{teacher}(x,k)\big).
$$

其中 `s_CLIP=exp(CLIP.logit_scale)`，只应用一次；`detach(teacher_probabilities)`，不能通过伪标签生成过程回传目标损失到 Base CLIP 或 Source Prompt。不要先计算两组概率再平均；应先融合文本相似度/logits，再做 softmax。无需对融合后的两组文本向量再次 L2 归一化。

**重要**：官方 `main.py` 的 multi-source 分支同时加入 `base`、`source_combined` 和多个 `source_specific` logits；这是多源路径。**本单源实验明确简化为 `base + 唯一 source` 各占 1/2**，避免对同一源域重复计权。把这一差异记录到最终报告。

### 4.2 Source CE 与 Target SPL

源域：

$$
 \mathcal L_S=\operatorname{CE}\left(s_{CLIP}\,Z_S F_S^\top,\;y_S\right).
$$

目标域 `Ti`：

$$
 \mathcal L_{T_i}=-\frac1{B_i}\sum_{j=1}^{B_i}\sum_{k=1}^{K}
 \hat p_{i,j,k}\log\operatorname{Softmax}\left(s_{CLIP}\,Z_{T_i}F_{T_i}^{\top}\right)_{j,k}.
$$

A/B 联合训练每步取 **一个 Source Batch + 每个 Target Domain 各一个 Target Batch**，总损失：

$$
 \boxed{\mathcal L_{A/B}=\mathcal L_S+\lambda_T\frac13\left(\mathcal L_{T_1}+\mathcal L_{T_2}+\mathcal L_{T_3}\right)}.
$$

C 的单 pair：

$$
 \boxed{\mathcal L_C=\mathcal L_S+\lambda_T\mathcal L_T}.
$$

`lambda_T` 默认 0.5（参照原论文）；三目标域等权，不依赖各域样本数量。一个联合 step 对**合成的总损失执行一次 optimizer.step()**；若为显存拆分反向，必须严格累加 `1/3` 的目标梯度，不能把它变成三个独立优化 step。

禁止加入：OT、OT 派生 Hard CE、额外 Hard Pseudo-label CE、熵最小化、对齐/对比/重放/目标伪标签阈值。训练阶段不使用目标图像的真实类别标签。可记录伪标签置信度/分布等**无标签诊断指标**。

## 5. 基于官方仓库的建议改造位置

先阅读代码、确认当前版本，再实现。下面是对已核对过的官方 `main` 分支的定位，不要求死守文件名：

| 官方文件 | 现状和需要处理的点 | 迁移建议 |
| --- | --- | --- |
| `model.py` | `PromptGenerator` 的 `ctx_cls/ctx_source/ctx_target/ctx_source_combined` 主要面向 1 target；`forward()` 返回 combined-source 与一个 target | 新建/重构 `MTDAPromptGenerator`，支持 A/B/C 参数共享和按域取文本 prompt；去除多余 combined 分支 |
| `main.py` | `for target_name in domain_list` 是每次选择一个目标域、其余作为多个源域；`training_mode` 为 `multi-source/source-combined` | 新建 `train_mtda.py` 作为入口，显式 `source_domain`、`target_domains` 与 `variant`；**不能只把 source/target 列表互换** |
| `main.py` | `running_means/running_count` 索引为 source×class；multi-source path 计算距离及权重 | 封装 `SourceCentroidBank`，单源积累，权重退化为 1；保留日志与测试 |
| `main.py` | Target 伪标签加了 base/source-combined/多源分支；存在 OT 开关、伴随 Hard CE | 提取 `compute_spl_soft_targets`，单源只融合 base + source；新训练入口不调用 OT |
| `dataloader.py` | `make_dataset()` 用目标真实标签统计 pseudo-label accuracy；`load_data()` 评估开启 shuffle/drop_last | 新增**无标签训练 Dataset/Loader** 与独立**有标签仅评估 Loader**；准确率按实际样本数统计，评估禁止 drop_last |
| `dataset.py` | `SingleSourceDataset/MultiSourceDataset` 从按类别命名目录返回 `(image,label[,domain])` | 源域沿用真实标签；目标训练显式丢弃类别标签且不暴露给训练函数，仅保留已知域 ID |
| `main.py` | `args_update()` 会强制覆盖训练步数；`test()` 使用固定 batch_size 累积总数；最终 `target_features` 缓存可能尚未写入 | 新入口尊重用户指定步数；纠正评估；从 checkpoint 的当前 Target Prompt **重算**文本原型，避免空缓存/过期缓存 |
| `samplers.py` | 原 `RandomDomainSampler` 适用于多源域 | 单源正常 shuffle；目标域各自独立 Loader/Iterator，遍历耗尽时仅在 `StopIteration` 后循环重启 |

建议目录组织（允许 Codex 按实现简化）：

```text
train_mtda.py                      # 参数解析、A/B/C 运行调度
mtda/
  prompts.py                      # A/B/C 的 prompt 构建与参数共享
  source_bank.py                  # 单源类质心运行均值
  spl.py                          # 仅 Base + Source 的 soft teacher
  data.py                         # 有标签源/无标签目标/独立评估
  trainer.py                      # joint 与 pairwise loss + step
  evaluate.py                     # target-only 评估与指标汇总
  experiment.py                   # 配置、seed、checkpoint、汇总
scripts/run_officehome_mtda.py     # 完整实验矩阵执行器
scripts/summarize_mtda.py         # 读取所有 run 生成 CSV/Markdown
 tests/                            # 至少覆盖下述单元测试
```

可按需要减少文件数量，但要保持可读、可测；尽量只做局部小改，不牵动 `main_PGA.py`、corruption 分支以及原 MSDA 实验。不要为迁移引入额外大框架。

## 6. 数据协议、默认训练配置与公平对照

### Office-Home 实验矩阵

1. 自动识别并检查 Office-Home 四个域（Art、Clipart、Product、Real World，具体目录名按用户提供数据集实际结构识别，不硬编码有歧义的拼写）。
2. 四个域轮流充当源域 `S`；每次剩余三个域 **同时**作为 A/B 的目标域；C 则对应三个独立 Source→Target runs。
3. 每个 seed 共 **A: 4 次联合训练；B: 4 次联合训练；C: 12 次 pairwise 训练，共 20 个 training runs**，产出 12 个有向适配任务 × 3 种方案的准确率。
4. **先完成 1 个固定 seed 的全部实验矩阵**。如果算力允许，补充多 seed（建议至少 3 个）并报告均值与标准差。重复运行不得覆盖已有结果。

### 数据隔离与评估协议

- 源域训练 Loader 提供 `(image, class_label)`；目标域训练 Loader 只提供 `(image, domain_id)` 或仅 `image`，其类别标签对 Trainer 不可见。目录组织中存在类名文件夹时，仅用于扫描样本路径，不能调用路径推断类别参与训练。
- 类别名称及 index 顺序从**源域标签集合**确定，确保 Base Prompt、Source Prompt、所有 Target Prompt 同一类别索引。训练前校验域目录和类别结构的一致性；不使用目标真实标签推断权重或损失。
- 目标域有标签 Loader 只在 `evaluate.py` 中实例化与迭代；评估时 `model.eval()`、`torch.no_grad()`、`shuffle=False`、`drop_last=False`。
- 默认沿用官方 Office-Home **transductive UDA** 协议：目标训练阶段可以看到目标域无标签图像，评估时同域样本以标签计算准确率；在报告中明确标注此协议。如果用户提供独立 train/test 划分，优先使用明确划分并在所有实验统一。
- 最终推理仅使用**对应目标域自己的 Target Prompt**，不能混合 Source/其他 Target Prompt 作为预测器，也不能用目标评估准确率选择最优参数。

### 训练默认值

- `M1=16, M2=16`；CLIP backbone 使用用户提供或原仓库支持的模型，所有方案一致；特征维度在运行时推断，避免硬编码。
- `steps=1000`（Office-Home 原仓库默认覆盖后的训练次数）；`batch_size` 默认沿用官方入口默认值，允许用户修改；`lambda_T=0.5`。
- 优先沿用**官方源码的 AdamW 与基础学习率**（`main.py` 默认 `prompt_learning_rate=0.003`）。原论文附录描述的是 SGD/0.005，与提交代码不同，**以源码为迁移起点，在报告写明选用哪套配置**。
- 使用 cosine 学习率调度时，明确其调用频率：建议按每次参数更新推进一次，以 `T_max=steps` 对齐；不要照搬原代码按 `steps/20` 的条件推进但仍设 `T_max=steps` 的不一致实现。该工程修正须记录，A/B/C 使用相同策略。
- 所有方案使用相同随机种子、每域 Batch Size、每目标域训练步数、CLIP 预处理、优化器、调度器和评估时间点。A/B 每步从三个目标域各取一个 Batch，C 每个 pair 每步取一个目标 Batch；即**每目标域看到的迭代数一致**，但 C 会重复训练源域教师 3 次，这个计算开销与共享程度差异要在报告说明。
- 评估与报告使用**固定最后一步 checkpoint** 的结果作为主要对比。可记录中途评估曲线，但不得凭目标真实标签挑选 best model 来作为主要指标。
- 训练损失权重、teacher logits 的融合比例在 A/B/C 完全相同。

## 7. 开发步骤与逐阶段验收

### Phase 0 — Clone、基线审计、环境确认

- clone、记录 commit SHA、创建开发分支；列出依赖与 CLIP checkpoint 加载方式。
- 运行 `--help`，确认参数入口；检查数据集目录及标签/域映射（用户提供路径后执行）。
- 写一份 `docs/implementation_notes.md`，列出“官方逻辑 → MTDA 替换逻辑”的具体映射，包括 `source_combined`、伪标签计算、Bank、评估 Bug、初始化、学习率调度的处理。
- 如果环境依赖不兼容，以**最小范围**兼容性修复为主；记录版本。无 GPU/权重时先推进 CPU/合成数据单测。

### Phase 1 — 数据与 Prompt 模块

- 实现独立无标签 Target Dataset、单源 Source Dataset 复用、分目标域 Loader。
- 实现 A/B/C Prompt 构造和可训练参数列表，支持 `forward_text(domain)`、`forward_source_text()`、`forward_target_text(target_id)`。
- 支持初始化快照与 checkpoint round-trip（保存/恢复后同一图片 logits 一致）。
- 增加参数共享、形状、梯度独立性测试。

### Phase 2 — Bank 与 SPL

- 实现单源 Running Mean Bank、计数与 resume。
- 实现 `teacher_logits = 0.5 * base_logits + 0.5 * source_logits`，`softmax` 后 `detach`。
- 实现源域 CE 与各目标域 SPL；A/B 联合单次 optimizer step；C pairwise step。
- 使用固定张量构造的单测验证 soft targets 和 loss 与数学公式一致。

### Phase 3 — 运行与评估

- 先做一个源域、一个目标域、2–5 step 的 smoke test；再做一个源域配三个目标域的 A/B joint smoke test，最后测试 C 独立启动。
- 检查所有 loss finite、梯度只落到允许的 Prompt、Bank 计数增加、checkpoint 能重载、准确率在 `[0,1]`。
- 完成 Office-Home **4 个源域 × A/B/C 全矩阵**（至少 1 seed）。不得因为运行耗时就自动只跑某个源域后宣称全矩阵完成；超出资源/时间限制时，保留已完成 run，列出剩余命令。
- 支持单 run 与批量矩阵两种操作，支持失败 run 重试/跳过已成功 run，并可从 checkpoint 恢复（恢复时必须包含 optimizer、scheduler、Bank、RNG 状态）。

### Phase 4 — 汇总与交付

- 输出可重跑的配置文件、每个 run 的终轮 checkpoint、逐步 loss 日志和评估结果。
- 自动汇总各 source→target task 的 A/B/C Accuracy；按源域求三个 Target 的宏平均，再计算所有 12 个任务的宏平均。
- 对比 CLIP zero-shot（不增加新的训练方法）作为参考；可选另记录训练完成的 Source Prompt 单独推理结果，但**三方案主表一律使用 Target Prompt**。
- 报告单个方案的时间/资源占用以及 A/B/C 之间的可解释差异；特别确认 Source Bank 在单源下仅做统计而不提供差异化 SPL 权重。

## 8. 自动化测试最低清单（必须通过）

- [ ] 数据接口：Trainer 完全拿不到目标域真实 `class_label`；来源域标签、目标域已知 `domain_id` 不混淆。
- [ ] 类别对齐：65 类的名称和 index 顺序在 source/base/每个 target 上相同；不匹配时 fail fast。
- [ ] Prompt 形状：Class `[K,M1,d_ctx]`，Domain `[M2,d_ctx]`，Encoded Text `[K,D]`。
- [ ] A：Source/全部 Targets 引用的是**同一个** Class Parameter；梯度由源与各目标损失共同贡献。
- [ ] B：`P_cls,S` 和三个 `P_cls,Ti` 是不同对象；源域 CE 不更新目标 Class，`Ti` 的 SPL 不更新其他目标 Class 或 Source Class。
- [ ] C：每个 pair 重新初始化且互不污染；pair 内 Source/Target 使用同一 Class Parameter。
- [ ] 初始化：同 seed 的起始 Class 数值在 A/B/C 一致，之后按各自共享规则更新。
- [ ] Bank：全部零初始化；对给定带标签特征流，累计均值与手工算术均值相等；缺失类别安全处理。
- [ ] 单源退化：`w=1`；改变质心数值不应改变当前 SPL soft target（只要 Source Prompt 不变）。
- [ ] SPL：先 `0.5 * logits_base + 0.5 * logits_source`，再乘 CLIP scale 做 softmax；对 teacher `detach`，只算 Soft CE。
- [ ] 训练：冻结 CLIP；无 OT/Hard CE/entropy 等额外损失；A/B 三个 Target Loss 等权取均值。
- [ ] 评估：只用对应 Target Prompt；完整遍历真实样本数（包含最后一个不满 Batch 的尾批）；读取 checkpoint 后重算文本特征。
- [ ] 输出：每 run 可重现、训练配置完整、汇总 CSV 的 run 数和 12 个 task 键完整。

## 9. 输出结构与最终实验报告

建议使用相对项目目录（下面均是**产物相对位置**，不是数据或权重路径）：

```text
runs/mtda_spl/
  A/<source>/<seed>/
  B/<source>/<seed>/
  C/<source>/<target>/<seed>/
    config.json
    metrics.json
    history.csv
    last.pt
    train.log
reports/
  officehome_mtda_A_B_C.csv
  officehome_mtda_summary.csv
  officehome_mtda_report.md
  implementation_notes.md
```

**逐 task 结果表**（每行一个 source→target，共 12 行；多 seed 时先逐 seed 明细，再聚合）：

| Source | Target | CLIP zero-shot | A Acc (%) | B Acc (%) | C Acc (%) | A−C | B−C |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 自动生成 | 自动生成 | … | … | … | … | … | … |

**汇总表**需展示：四个 Source 分别的 A/B/C 三目标平均准确率、全部 12 task 的宏平均（以及多 seed 的标准差，如果执行），每个变体的训练耗时与最高显存。不得把 A/B 每次联合训练的三目标结果只记录为一个总体数字。

`officehome_mtda_report.md` 必须包括：

1. 环境、官方 upstream commit SHA、实际配置、随机种子、数据协议；
2. A/B/C 训练细节与源码改动摘要；
3. 每个目标域准确率、每 Source 平均、总 Macro Average，按需附训练曲线；
4. 单源 Bank 覆盖率及其 SPL 权重退化为 1 的说明；
5. A 对 B/C 的收益、B 与 C 是否接近、运行耗时差异；
6. 遇到的失败、修复、未完成实验，以及**实际执行过的命令**；
7. 将后续可能的 MTDA 改良单列为 **Future Work**，本阶段**不要实现**。

## 10. 推荐命令接口（只规定形式；实际路径由用户提供）

```bash
# 从官方源码开展工作
git clone https://github.com/VuongLong/Clustering-Reinforcement-Prompt-Learning.git
cd Clustering-Reinforcement-Prompt-Learning
git rev-parse HEAD
git switch -c feat/mtda-spl-baseline

# 新入口需具备清晰的 --help；具体数据与权重参数在运行时由用户告知
python train_mtda.py --help

# 运行单个配置，指定一个 source / variant / seed，A/B 自动处理剩余全部 target；C 分别运行三个 pair
python train_mtda.py --variant A --source "<SOURCE>" --seed 42 [用户提供的数据和权重参数]
python train_mtda.py --variant B --source "<SOURCE>" --seed 42 [用户提供的数据和权重参数]
python train_mtda.py --variant C --source "<SOURCE>" --target "<TARGET>" --seed 42 [用户提供的数据和权重参数]

# 整体矩阵与汇总（实际 CLI 可调整，需在 README 写明）
python scripts/run_officehome_mtda.py --help
python scripts/summarize_mtda.py --runs-dir runs/mtda_spl
```

方括号中文提示是**占位说明，不可原样复制执行**；Codex 应提供真实且可复现的 CLI、README 示例以及缺失参数时的错误提示，不得擅自编造本地路径。

## 11. 完成定义（Definition of Done）

1. 基于官方仓库已实现 A/B/C 三种定义准确、可运行的 MTDA/SPL-only 代码；原 MSDA 代码保留可辨认。
2. 所有最低自动化测试通过，能够从日志与参数列表证明 CLIP 冻结、Prompt 共享关系正确、没有目标训练标签泄漏。
3. 完成至少一个 seed 的 Office-Home 全矩阵（4 + 4 + 12 个训练 run，12 个有向任务 × 3 方案）；若因外部条件未完成，要明确列出**完成数量、阻塞和剩余任务**。
4. 生成完整 CSV、Markdown 报告、实际运行配置、checkpoint、日志和环境/代码版本信息；报告可准确回答“原始 SPL Prompt 设计在 MTDA 上是否有效，以及 A/B/C 哪个更好”。
5. **不实现新算法**。关于单源质心 Bank 失去源域加权能力的观察只写入分析和 Future Work。

---

**执行优先级：**先可运行性与实验正确性，再跑全矩阵，最后做代码整理与报告。不要用未经证实的性能预期代替真实 Office-Home 结果。
