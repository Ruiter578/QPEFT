# Final Report v7 检查版：全文中文翻译、写作思路与 Review 报告

**对象文件：** `Final_Report_QPEFT_revised_v7_tables_polished检查版.docx`  
**项目题目：** Progressive Co-Optimization for Vision Transformer Quantization with Parameter-Efficient Fine-Tuning  
**版本定位：** 本文档是当前 v7 检查版的中文对照、写作逻辑说明和最终阶段 review 报告。它用于帮助人工检查英文正文是否准确、自然、符合学校格式和 AI/CV 顶会式技术写作要求。  
**重要说明：** 本文档是中文审稿与理解文件，不是要提交的英文正文。最终提交仍以英文 Word 报告为准。

---

## 0. 总体 Review 结论

当前 v7 检查版已经基本进入可用主版本状态。与早期版本相比，它已经解决了若干高风险问题：References 之后的 Appendix 模板残留已删除；正文中未发现 `uploaded spreadsheets`、Excel 文件名、GPT 项目文件名、Codex、placeholder、planned、reserved、Appendix 等明显内部过程信息；Results 的 4.1 已从内部证据来源说明改为正式 evaluation protocol；Table 1–9 已转为更接近论文风格的三线表；Algorithm 已通过 LaTeX 裁剪图片方案独立解决；引用体系与 31 条 Master Reference Registry 对齐。

当前剩余主要问题不是正文大改，而是最终装配：Figure 1、Figure 4、Figure 5、Figure 6 尚需高质量插入；Figure 2/3 如无真实数据应删除；Contents、List of Figures、List of Tables 插图后必须更新；图表跨页和视觉效果仍需在 Microsoft Word 与最终 PDF 中逐页检查。

---

## 1. Abstract 中文翻译

本报告研究在有限下游数据、受限可训练参数和严重量化噪声共同约束下，Vision Transformer（ViT，视觉 Transformer）的低比特适配问题。项目提出 Quantized Parameter-Efficient Fine-Tuning（QPEFT，量化参数高效微调），这是一种基于 fake quantization（伪量化）的适配框架。该框架冻结大部分预训练 ViT backbone（骨干网络）参数，同时联合优化 Low-Rank Adaptation（LoRA，低秩适应）模块、任务分类头和选定的量化相关参数。实现上，该方法作用于 attention（注意力）和 feed-forward（前馈网络）中的线性投影，引入低比特权重和激活量化器，并对 post-softmax（Softmax 后）和 post-GELU（GELU 后）张量采用自适应激活处理。

为避免从全精度适配到低比特模拟的突然转换，框架引入 Stochastic Element-wise Exposure to Quantization（SEEQ，随机逐元素量化暴露），在训练过程中逐步让张量元素暴露于低比特反量化值。实验评估使用 ViT-B/16 和 ViT-L/16 backbone，在 VTAB-style 分类任务上进行，同时对比 post-training quantization（PTQ，后训练量化）和 QAT-style baseline，并补充 full-image classification 结果以及 shift 与 stochastic exposure schedule 消融。评估结果表明，QPEFT 在 3-bit 和 4-bit 伪量化下保持了具有竞争力的性能。ViT-B/16 在 3-bit 和 4-bit 下的 Tot. Avg. 分别为 69.95 和 71.21；ViT-L/16 分别达到 72.99 和 73.68，其中 4-bit 设置接近其 FP32 LoRA reference 的 73.71。结果说明，量化和参数高效适配应被视为耦合优化问题，而不是彼此独立的压缩和微调阶段。本研究的范围限定在伪量化和分类导向评估，不声称真实 integer-kernel deployment（整数核部署）或 dense prediction（密集预测）上的普遍优势。

**Keywords 中文对照：** Vision Transformer（视觉 Transformer）；low-bit quantization（低比特量化）；parameter-efficient fine-tuning（参数高效微调）；Low-Rank Adaptation（低秩适应）；fake quantization（伪量化）；quantization-aware adaptation（量化感知适配）；post-training quantization（后训练量化）；VTAB-style evaluation（VTAB 风格评估）。

### 写作思路

摘要采用“问题约束—方法结构—机制组件—实验范围—关键结果—结论边界”的顺序。它既给出核心数字，又明确 fake quantization 和 classification-oriented evaluation 的边界，避免把项目结果包装成真实硬件部署或通用 SOTA。

---

## 2. Acknowledgements 中文翻译

我谨向我的导师表达诚挚感谢。导师在整个毕业设计过程中给予了指导、耐心和关键反馈，帮助我不断收敛研究方向、保持清晰的技术边界，并将代码实现工作与连贯的学术论证联系起来。我也非常感谢课题组博士生师兄在日常研究中的支持。他对代码库、实验流程和 Vision Transformer 量化实际困难的讲解，对我从初始概念方案推进到可复现项目和具体实验证据起到了重要作用。他在调试、实验理解和研究建议方面的帮助，是本项目能够完成的重要基础。我也感谢课题组成员在模型压缩、参数高效微调和实验报告方面的讨论。最后，我感谢格拉斯哥学院、电子科技大学以及家人在本报告完成期间给予的支持，使我能够集中精力完成技术实现与项目写作。

### 写作思路

致谢语气正式、自然、克制。它感谢导师、博士生师兄、课题组、学校和家人，符合工程 final report 的致谢规范，同时没有过度煽情。

---

## 3. Introduction 中文翻译与写作思路

### 3.1 1.1 Background and Motivation 中文对照

Transformer 架构由于 self-attention（自注意力）能够在不依赖 recurrence（循环结构）或固定 convolutional locality（卷积局部性）的情况下建模长程依赖，已成为现代人工智能中的核心模型族。Vision Transformer（ViT，视觉 Transformer）将这一思想扩展到计算机视觉，通过把图像划分为 patch tokens（图像块标记）并输入 Transformer encoder（Transformer 编码器）进行处理。这种形式将视觉识别统一为 sequence modeling（序列建模）问题，支持大规模预训练，并能有效迁移到下游分类和密集预测任务。DeiT 等 data-efficient training（数据高效训练）策略表明，在比原始 ViT 设置更实际的数据条件下，Transformer-based visual backbones（基于 Transformer 的视觉骨干网络）仍然可以训练和适配。Swin Transformer 这类层级结构也表明，Transformer 设计可扩展到 multi-scale visual recognition（多尺度视觉识别），同时保留 attention-based contextual modeling（基于注意力的上下文建模）优势。

ViT 的实际成功也带来了部署问题。标准 ViT block 包含 multi-head self-attention（多头自注意力）、feed-forward multilayer perceptrons（前馈多层感知机）、residual connections（残差连接）和 normalization layers（归一化层）。Attention 相对于 token sequence length（标记序列长度）具有二次复杂度，而 attention 和 feed-forward 模块中的线性投影主导了参数存储和算术开销。这些性质在服务器端研究设置中可以接受，但在边缘设备、嵌入式平台和大规模云服务中不理想，因为这些场景更关注 memory footprint（内存占用）、energy consumption（能耗）、latency（延迟）和 concurrent inference cost（并发推理成本）。即使目标系统不是小型设备，降低数值精度也可以减少存储和带宽压力，这对于重复适配和部署大型预训练 backbone 尤其重要。

Model quantization（模型量化）通过使用低比特数值格式表示 weights（权重）和 activations（激活）来降低成本。经典 integer-oriented quantization（面向整数推理的量化）把连续张量映射到离散网格，使模型在导出到兼容 runtime 后可以获得更低存储成本和整数运算潜在收益。在研究实现中，fake quantization（伪量化）常用于训练期间模拟低比特权重和激活的前向影响，但许多操作仍由浮点 kernel 执行。本报告强调这一点：项目研究的是基于伪量化的适配与优化稳定性，而不是已经完成硬件部署或测得整数核加速。

低比特量化对 ViT 特别有吸引力，因为提升表征能力的结构也带来了显著部署成本。然而，ViT 量化不是卷积神经网络量化的直接延伸。LayerNorm 输出可能具有严重通道间差异，post-softmax attention values 高度非均匀，post-GELU activations 可能呈现非对称或类幂律分布。因此，当 bit-width 降至 4-bit 或 3-bit 时，ViT 往往比传统卷积 backbone 出现更明显性能下降。已有 ViT-specific PTQ 方法通过 scale reparameterization（尺度重参数化）、twin/logarithmic quantizers（双均匀或对数量化器）、reconstruction losses（重构损失）和 activation-specific treatment（激活专门处理）来应对这些现象。这些工作为本研究提供动机，同时也说明：在 ultra-low bit-width（超低比特）、limited adaptation data（有限适配数据）和 trainable downstream behavior（可训练下游行为）共同存在时，单纯量化并不足够。

Parameter-Efficient Fine-Tuning（PEFT，参数高效微调）提供了互补路径。PEFT 不更新全部预训练参数，而是通过训练少量参数或附加轻量模块来适配模型。Low-Rank Adaptation（LoRA，低秩适应）是一种代表性重参数化 PEFT 方法，它冻结预训练权重，并向选定线性层注入可训练低秩更新。对于视觉模型，PEFT 可以减少可训练参数数量，并限制小数据迁移中的过拟合。但 PEFT 本身不能解决超低比特量化带来的数值不匹配。一个只在全精度计算路径中训练的 LoRA adapter 可能能补偿下游任务偏移，但当权重、激活和注意力张量被量化时，它未必保持鲁棒。

本项目正是由这两种压力的交汇所驱动。ViT 部署需要压缩和低比特模拟，而有限数据下游适配需要轻量训练机制。把量化和 PEFT 视为独立步骤并不一定最优，因为 adapter 是在某一数值环境下学习的，而 quantizer 会改变该环境。因此，本报告的核心动机是探索一种 progressive co-optimization（渐进式协同优化）策略，使量化参数、激活量化行为和基于 LoRA 的适配在受控 exposure schedule（暴露调度）下共同优化。Figure 1 展示从预训练 ViT 到基于 LoRA 的低比特适配模型的总体流程。

### 3.2 1.2 Problem Statement and Key Challenges 中文对照

本项目研究的问题是：如何在有限数据条件下，通过联合优化量化与参数高效微调，稳定超低比特 Vision Transformer 适配。目标设置包含预训练 ViT backbone、VTAB-style classification adaptation（VTAB 风格分类适配）、用于权重和激活的低比特伪量化，以及基于 LoRA 的可训练更新。实际问题不仅是能否得到一个低比特 ViT，而是当大部分 backbone 参数被冻结、只有少量参数可训练时，适配是否仍然稳定。这些可训练参数包括 LoRA 模块、分类头和选定量化相关变量。

PTQ 具有吸引力，因为它可以用少量无标签数据校准或重构训练好的模型，并避免完整端到端 retraining（重新训练）。但在超低比特 ViT 设置中，这种效率也可能成为限制。QDrop 表明，activation quantization（激活量化）应在重构中被考虑，而且部分暴露于激活量化可能改善低比特 PTQ 的鲁棒性。ViT-oriented PTQ 研究进一步说明，post-LayerNorm、post-softmax 和 post-GELU 张量需要特殊处理，因为它们的分布不适合简单 uniform quantization（均匀量化）。这些观察说明，PTQ 高效但可能过于刚性，尤其是在下游任务需要超出校准范围的适配时。

QAT 通常能恢复更多精度，因为模型在训练前或训练中暴露于量化效应。然而，QAT 也引入优化困难。低比特 QAT 依赖 straight-through gradient approximations（直通梯度近似）处理不可导 rounding（舍入），已有工作表明量化权重可能在相邻量化格点之间 oscillate（振荡）。对于 ViT，这种振荡可能被 learnable scaling factors（可学习缩放因子）以及 self-attention 中 query/key 投影之间的交互进一步放大。该问题与本项目直接相关，因为当前实现使用可学习量化参数，并试图适配敏感 attention 和 feed-forward 组件，而不是仅校准静态范围。

第二个挑战是量化误差与低秩适配能力之间的相互作用。LoRA 假设任务特定更新可表示为添加到选定权重矩阵上的低秩残差。这个假设对于适配通常有效，但量化误差未必落在同一低秩子空间中。当 backbone 冻结时，LoRA 必须同时补偿下游任务偏移和低比特量化引入的数值扰动。如果量化突然施加，adapter 可能收到不稳定梯度，并学习到只适用于瞬时数值状态的修正。反过来，如果 adapter 完全在全精度路径训练，之后再量化，则学习到的更新可能与低比特前向路径不兼容。

第三个挑战是 ViT 内部激活异质性。用单一固定策略量化普通线性层激活、post-GELU activations 和 post-softmax attention probabilities 会产生不匹配误差。已有 ViT PTQ 方法使用 adaptive logarithmic quantization（自适应对数量化）、scale reparameterization 或 reconstruction-based losses 来降低这些不匹配。在可训练 QPEFT 设置中，挑战是保持这种分布感知处理，同时使方法足够轻量，适合有限数据和毕业设计约束。

最后，项目需要清晰技术范围。报告评估的是分类导向 Vision Transformer 适配中的 LoRA-based QPEFT 实现，不提出关于 dense prediction 或 integer-kernel deployment 的一般性声明。Table 1 概括 PTQ、QAT、PEFT 和 QPEFT 在这一研究空缺中的定位。

### 3.3 1.3 Overview of the Proposed Framework and Contributions 中文对照

报告提出 QPEFT 作为低比特 ViT 适配的 progressive co-optimization framework（渐进式协同优化框架）。该框架结合冻结的预训练 ViT backbone、LoRA-based trainable updates（基于 LoRA 的可训练更新）、权重和激活低比特伪量化器、可学习量化参数、自适应激活量化和 SEEQ。当前实现是 LoRA-based QPEFT，而不是 DiLoRA-based QPEFT。LoRA 模块插入选定 Transformer 线性层，包括注意力投影和前馈层，同时大部分 backbone 保持冻结。

核心思想是避免把适配和量化当作顺序且独立的两个过程。顺序流程可能先适配全精度模型再量化，或先构建量化模型再添加 adapter，这两种方式都会产生 mismatch（不匹配）。QPEFT 则在同一个优化过程中训练任务 adapter 和量化行为。LoRA 分支学习任务特定修正，量化分支学习或更新定义低比特模拟的参数。

SEEQ 是 progressive exposure mechanism（渐进暴露机制）。设 $F_i$ 表示第 $i$ 个张量元素的全精度值，$Q_i$ 表示量化再反量化后的对应低比特值。SEEQ 在元素级别采样前向路径使用的值。训练早期较高的 full-precision retention probability（全精度保留概率）可以保持优化表面更平滑；训练后期较高的 quantization exposure probability（量化暴露概率）可以让模型更接近低比特路径。代码实现层面，这与 DropSTE-style stochastic exposure 相关，即训练时随机保留连续值或暴露量化值。

本报告贡献包括：第一，将低比特 ViT 适配表述为量化与参数高效微调之间的协同优化问题；第二，实现 LoRA-based QPEFT pipeline，在 3-bit 或 4-bit 伪量化下冻结大部分 backbone 并联合训练 LoRA、分类头和量化相关参数；第三，提出 SEEQ 作为全精度张量值和低比特反量化张量值之间的渐进随机暴露机制；第四，建立用于 VTAB-style 分类任务低比特 QPEFT 评估的可复现实验基础。

### 3.4 1.4 Scope and Claim Boundary 中文对照

本报告范围是 Vision Transformer 在分类迁移场景下的低比特伪量化适配。报告区分 simulated quantization（模拟量化）和 deployable integer inference（可部署整数推理）。Fake quantization 在前向传播中插入量化和反量化行为，使训练经历低比特扰动，但 PyTorch 执行仍可能依赖浮点操作。因此项目不声称真实硬件 speedup、power reduction 或 memory-bandwidth reduction。

报告也区分 LoRA-based QPEFT 与 QLoRA-style large language model fine-tuning。QLoRA 是用于量化大语言模型高效微调的特定协议；本项目在 Vision Transformer 中使用 LoRA-like low-rank adaptation，并结合伪量化和可学习量化行为。因此应描述为 LoRA-based QPEFT 或 ViT 的 quantization-aware LoRA adaptation，而不是 QLoRA。

严格来说，本研究不应被描述为标准 PTQ 或标准 QAT。它不是 PTQ，因为它不仅仅在训练后校准冻结模型；它在下游适配中引入可训练 LoRA 参数、分类头参数和量化相关参数。它也不是常规 full-model QAT，因为大部分预训练 backbone 权重被冻结，目标是在伪量化下进行参数高效下游适配，而不是端到端重训练整个模型用于整数部署。

### 写作思路与 Review

Introduction 逻辑完整，层次清楚：从 ViT 部署成本到量化困难，再到 PEFT 的适配优势与局限，最后引出 QPEFT。当前版本没有发现明显项目内部文件泄漏，也没有把方法夸大为 SOTA 或硬件加速。保守边界存在，但总体已不像内部证据审计，属于可接受的正式论文式 claim boundary。

---

## 4. Related Work 中文翻译与写作思路

### 4.1 2.1 Vision Transformers and Efficient Visual Adaptation

Transformer 最初作为以 self-attention 为核心的序列模型建立，通过学习 token 之间的交互建模上下文关系，而不是依赖循环结构或固定卷积局部性。ViT 将这一形式转移到图像中，把图像表示为 patch token 序列，并通过 Transformer encoder blocks 处理。这一转变不仅挑战了卷积神经网络在视觉识别中的主导地位，也使视觉建模越来越依赖大规模线性投影、注意力图、归一化层和 token-wise feed-forward networks。这些组件正是本项目研究量化问题的核心，因为它们的数值分布和相互作用不同于传统卷积 backbone。

后续 ViT 变体让该架构更适合实际视觉学习。DeiT 说明，在知识蒸馏和精心训练策略帮助下，图像 Transformer 可以在更实际的数据条件下训练。Swin Transformer 引入层级特征图和 shifted local windows，降低全局注意力成本并提升密集视觉任务适用性。这些发展表明，efficient visual adaptation 不只是减少参数，还需要在任务约束下保留 token interactions 和 multi-scale feature behavior。对于低比特适配，这一点尤为关键，因为激进量化可能破坏赋予 ViT 迁移能力的 attention 和 feed-forward 结构。

### 4.2 2.2 Post-Training Quantization for Vision Transformers

PTQ 具有吸引力，因为它可以使用有限校准数据压缩预训练模型，避免完整端到端重训练。AdaRound 将 rounding 转化为数据感知优化问题，说明最近邻舍入未必最优。BRECQ 等 block reconstruction 方法将 PTQ 从局部张量校准推进到 layer/block output matching。QDrop 进一步指出，在极低比特 PTQ 中激活量化应被纳入重构，但部分激活量化暴露可能改善 flatness 与泛化。这些思想与本项目相关，因为 SEEQ 也使用 stochastic exposure，但 QDrop 仍是 PTQ reconstruction 方法，而 QPEFT 使用下游参数高效优化。

直接把普通 PTQ 应用于 ViT 很困难，因为 ViT 组件会产生不适合简单均匀量化的分布。早期 ViT PTQ 工作识别了 post-LayerNorm、post-softmax 和 attention-related tensors 的量化困难。PTQ4ViT 使用 twin uniform quantization 和 Hessian-guided calibration 处理 post-softmax 和 post-GELU 激活。RepQ-ViT 通过 scale reparameterization 将准确性导向的量化设计和推理友好实现解耦。AdaLog 则强调对 post-softmax 和 post-GELU 的类幂律分布进行自适应非均匀处理。APHQ-ViT、FIMA-Q 和 I&S-ViT 等进一步说明，超低比特 ViT PTQ 不是简单 scale selection 问题，而涉及重构稳定性、激活处理和优化几何。

PTQ 对本项目的结构性限制在于：其主要假设是预训练模型可通过有限更新被量化和校准。当目标设置还要求有限数据下游适配时，calibration alone 不提供任务特定补偿机制。因此需要把低比特量化和轻量可训练适配一起优化。

### 4.3 2.3 Quantization-Aware Training for Low-Bit Vision Transformers

QAT 通过在训练或微调中暴露量化效应来克服 PTQ 弱点。经典整数推理导向量化工作建立了 fake quantization 在训练中准备低精度推理的基本思路。LSQ 让 quantization step size 可学习，这与本项目中的可学习量化参数密切相关。原则上，QAT 比 PTQ 更强，因为模型可以通过梯度优化适应量化噪声。但实际中，QAT 带来更高训练成本、对超参数敏感以及低比特不稳定问题。

Q-ViT 指出，在低比特 ViT 中，注意力图的信息失真是主要问题之一，并提出 information rectification 与 distribution-guided distillation。Oscillation-Free Quantization 关注低比特 ViT 中的 weight oscillation（权重振荡），指出 learnable scale 与 query/key 投影交互可能加剧不稳定。这些工作说明，QAT 不能简单等同于“训练中加量化”。它必须处理优化稳定性、注意力信息损失和可学习量化参数之间的相互作用。

本项目更接近 quantization-aware PEFT，而不是完整 QAT：量化效应进入下游优化，但训练只发生在 LoRA、分类头和部分量化相关参数上，大部分 backbone 冻结。

### 4.4 2.4 Parameter-Efficient Fine-Tuning for Vision Models

PEFT 通过训练少量参数或附加模块降低大模型适配成本。LoRA 将任务更新表示为低秩矩阵分解，冻结原始权重，只训练低秩增量。Visual Prompt Tuning、AdaptFormer 和 Sparse-Tuning 等方法说明，视觉模型的高效适配可以通过 prompt、adapter 或稀疏更新实现。这些方法对本项目很重要，因为低比特适配通常处于有限数据和有限算力环境，完整微调可能成本高且容易过拟合。

但 PEFT 本身不保证低比特鲁棒性。QLoRA 说明量化与低秩适配可以结合，但它主要面向大语言模型；本项目关注 Vision Transformer 中的 LoRA-based QPEFT。DoRA 等方法可作为未来扩展，但不能当作当前实验结果来源。

### 4.5 2.5 Research Gap and Positioning

Related Work 最终收束到一个 coupling problem：ViT 提供强迁移 backbone，但量化敏感；PTQ 高效但适配能力有限；QAT 能处理量化噪声但训练成本高且可能不稳定；PEFT 降低训练参数但不天然解决低比特扰动。因此，本项目将量化器和 adapter 视为应共同优化的对象，而不是两个独立阶段。

### 写作思路与 Review

Related Work 质量较高，不是文献列表，而是按研究流派与技术瓶颈组织。Table 2 的作用是把文献流派与 QPEFT 缺口对应起来。当前主要注意事项是引用编号必须保持 registry 对齐，不要因为删除论文 PDF 而删除 registry 条目。

---

## 5. Materials and Methods 中文翻译与写作思路

### 5.1 3.1 Problem Formulation and Preliminaries

本章把问题形式化为：给定预训练 ViT backbone 和下游分类任务，在大部分 backbone 冻结的前提下，通过 LoRA、分类头和量化相关参数的有限训练，实现低比特伪量化环境下的稳定适配。Fake quantization 将连续张量映射为低比特离散值，再反量化回浮点形式用于前向模拟。LoRA 则把线性层权重更新表示为低秩增量。

### 5.2 3.2 Quantization Sensitivity Diagnosis

本节解释 ViT 量化敏感性来自 LayerNorm、Softmax、GELU、attention map 和 residual propagation 等结构。它说明 Figure 2 和 Figure 3 的诊断意义，但当前如果没有真实 sensitivity 或 activation dump 数据，应删除这两张图，不能用 AI 伪造。

### 5.3 3.3 Proposed QPEFT Framework

QPEFT 框架将 frozen ViT core、LoRA adapters、weight/activation quantizers、adaptive activation quantization 和 SEEQ 结合起来。LoRA 提供任务适配能力，量化器模拟低比特数值环境，SEEQ 控制训练中量化扰动的暴露强度。Figure 4 应作为核心方法图，展示 QPEFT 在 ViT block 内部如何作用于 qkv、proj、fc1、fc2 以及 post-softmax/post-GELU 位置。

### 5.4 3.4 SEEQ and DropSTE-style Stochastic Exposure

SEEQ 通过逐元素随机选择全精度值或低比特反量化值，使模型从较平滑的训练状态逐步过渡到低比特扰动环境。DropSTE-style exposure 是代码层面最接近这一思想的实现描述。写作重点是说明它是 training-time exposure mechanism，而不是硬件推理机制。

### 5.5 3.5 Quantization and Adaptation Modules

本节介绍 LoRA 注入位置、权重量化器、激活量化器和 attention softmax quantization。关键点是 LoRA 注入 qkv、proj、fc1、fc2；普通激活和 sensitive activations 使用不同处理；post-softmax 和 post-GELU 激活需要自适应或非均匀处理。

### 5.6 3.6 Training and Reparameterization Protocol

本节说明训练时冻结大部分 backbone，只训练 LoRA、分类头和 selected quantization-related parameters，并使用 AdamW、DropSTE-style rounding relaxation 和 checkpoint 选择。写作上应强调 protocol，而不是逐行代码说明。

### 5.7 3.7 Experimental Setup

当前 v7 版本已经将 3.7 改成正式实验设置：classification-oriented VTAB-style adaptation 是核心；ViT-B/16 和 ViT-L/16 是主要 backbone；结果包含 QPEFT、PTQ baseline、QAT-style baseline、full-image classification 补充结果、ablation 和 TensorBoard diagnostics。Table 3 的标题不再过短，用于概括分类导向适配和诊断评估。

### 5.8 3.8 Claim Scope and Validation Protocol

本节明确结果能支持和不能支持什么：可以支持低比特 fake quantization 下 QPEFT 的分类适配结果；不能支持真实 hardware acceleration、dense prediction generalization、multi-seed robustness 或 universal superiority。当前版本语气较为正式，不再像内部证据审计。

### 写作思路与 Review

Methods 的核心任务是“让读者复现并理解方法”，不是详细展开内部文件。当前 v7 的 3.7 和 3.8 相比旧版本已有明显改善。但最终仍应检查 Figure 2/3 是否会在无真实数据情况下保留，以及 Figure 4 是否足够清楚地区分 LoRA、quantizer 和 SEEQ。

---

## 6. Results 中文翻译与写作思路

### 6.1 4.1 Evaluation Protocol

Results 首先说明 evaluation protocol：主结果来自 VTAB-style 分类任务，任务按 Natural、Specialized 和 Structured 分组。Group Avg 表示三个组平均的非加权均值，Tot. Avg 表示 19 个任务准确率的直接平均。由于三个组包含的数据集数量不同，若 Tot. Avg 可用，则优先作为主 summary metric。该节还说明 QAT-style baselines、CIFAR/Food101/SVHN full-image classification 和 ablation 的作用，以及 TensorBoard case study 的诊断性质。

### 6.2 4.2 Main QPEFT Results on VTAB-style Tasks

Table 5 展示 ViT-B/16 和 ViT-L/16 的 QPEFT 主结果。ViT-B/16 的 3/3 和 4/4 QPEFT Tot. Avg 分别为 69.95 和 71.21；ViT-L/16 的 3/3 和 4/4 QPEFT 分别为 72.99 和 73.68。ViT-L/16 的 2/2 行明显较低，为 61.92，说明 2-bit 设置在当前分类协议下可靠性不足。

写作重点是：先报告事实，再解释趋势；不把 4-bit 接近 FP32 LoRA 扩大为通用结论。

### 6.3 4.3 Comparison with PTQ Baselines

Table 6 对比 QPEFT 和代表性 PTQ baselines，包括 AdaLog、RepQ-ViT、PTQ4ViT 和 QDrop。ViT-B/16 3-bit 下，QPEFT 为 69.95，QDrop 为 61.72，AdaLog 为 60.35。ViT-B/16 4-bit 下，QPEFT 为 71.21，AdaLog 为 70.16。ViT-L/16 上，QPEFT 在 3-bit 和 4-bit 下分别为 72.99 和 73.68，高于对应 AdaLog 数值。

写作重点是：这是 matched VTAB-style bit-width settings 下的对比，不是所有 PTQ 方法和所有设置下的 universal ranking。

### 6.4 4.4 Comparison with QAT Baselines and Full Image Classification Evidence

Table 7 将 QAT-style baseline 和 full-image classification 补充结果放在一起。QAT-style baseline 的 3-bit 和 4-bit Tot. Avg 分别为 52.79 和 57.83，低于对应 ViT-B/16 QPEFT 结果。CIFAR、Food101 和 SVHN 的 full-image classification 结果用于补充说明 QPEFT 在完整图像分类任务上也有稳定性迹象，但这与 VTAB-style 19-task protocol 不完全相同。

### 6.5 4.5 Ablation Study

Table 8 报告消融：post-GELU shift 从 no-shift 的 64.56 提升到使用 constant shift 0.17 的 69.84。drop_epc=05 达到 70.08，drop_epc=04 到 07 形成较稳定区间。这说明适度 stochastic exposure schedule 可能优于过早或过晚暴露。

写作重点是：用 “suggests” 或 “indicates” 解释趋势，不说 “proves”。

### 6.6 4.6 TensorBoard Case Study on CIFAR

Figure 5 和 Figure 6 用于 CIFAR case-study diagnostics。Figure 5 显示 validation Acc@1 trajectory；Figure 6 显示 representative gradient traces。它们帮助判断训练轨迹是否合理、LoRA 和 quantization-related parameters 是否获得非退化梯度。但它们不是 multi-task 或 multi-seed validation 的替代。

### 6.7 4.7 Parameter Efficiency and Summary of Findings

Table 9 总结可训练参数。ViT-B/16 QPEFT 的可训练组件为 165,888 + 83,484，总计 249,372；ViT-L/16 QPEFT 为 442,368 + 222,168，总计 664,536。这些是 trainable components 的算术汇总，不能直接推断 runtime efficiency、energy saving 或 hardware acceleration。

### 写作思路与 Review

Results 当前主线正确：先主结果，再 PTQ/QAT 对照，再 ablation，再 TensorBoard case study，最后参数效率。当前未发现内部文件名或中文。语气仍然比较克制，但已经可接受，因为这是一份工程 final report，保留 claim boundary 是合理的。

---

## 7. Discussion 中文翻译与写作思路

### 7.1 5.1 Interpretation of Main Findings

主结果说明 QPEFT 能在 3-bit 和 4-bit 伪量化下保留较强性能，尤其是 ViT-L/16 的 4-bit QPEFT 接近 FP32 LoRA reference。2-bit 明显下降，说明超低比特存在边界。Structured tasks 通常低于 Natural 和 Specialized tasks，说明结构化推理类任务更容易受到低比特扰动影响。

### 7.2 5.2 Relation to PTQ, QAT, and PEFT Baselines

PTQ 对比得出 nuanced conclusion（有条件结论）：强 PTQ 方法在某些 4-bit 设置下仍有竞争力，但许多 PTQ baseline 在 3-bit 下明显下降，而 QPEFT 通过低比特模拟和 LoRA-based adaptation 保留更高平均准确率。QAT-style baseline 低于 QPEFT，但这不证明 QPEFT 普遍优于所有 QAT；它只说明在当前数据、模型和训练配置下，没有 LoRA-style adaptation 的直接 QAT-style 优化更不稳定。

PEFT 关系也需要细致表述。FP32 LoRA 行是没有低比特约束下的参数高效适配参考；QPEFT 的价值不只是 LoRA 能适配 ViT，而是 LoRA 在 fake quantization 环境下与量化行为共同训练仍然有效。

### 7.3 5.3 What the Ablations Suggest

Shift ablation 说明 post-GELU activation handling 对稳定性重要。GELU 产生窄负区间和较宽正区间，低比特网格若对齐不好，会浪费表示层级或造成 clipping/rounding error。constant shift 可能改善激活分布与量化器的对齐。

drop_epc ablation 说明 progressive exposure schedule 不应过早或过晚。适中 schedule 使模型先在较平滑路径中适配，再逐渐面对低比特扰动，类似 curriculum learning。

### 7.4 5.4 Reliability and Reproducibility

本节讨论可靠性和可复现性。可复现性来自明确 backbone、bit-width、LoRA 注入位置、可训练参数、评估协议、TensorBoard diagnostics 和结果表。可靠性边界在于：并非多 seed 大规模重复验证，也未覆盖 dense prediction 和真实硬件部署。

### 7.5 5.5 Limitations and Threats to Validity

局限包括 single-seed risk、baseline fairness、fake quantization 与真实整数推理的差异、TensorBoard case study 的覆盖有限、ablation 语义仍需源代码核对，以及缺少 dense prediction validation。这些限制是可信论文必须承认的，不是缺陷遮掩。

### 写作思路与 Review

Discussion 当前质量较高，能把 Results 的数值与方法机制联系起来，同时避免过度 claim。局部仍可稍微减少 “evidence” 一词频率，但总体已不像项目内部审计。该章适合保留。

---

## 8. Conclusions and Future Work 中文翻译与写作思路

### 8.1 6.1 Conclusions

本项目研究了有限数据和有限可训练参数预算下的低比特 ViT 适配问题。报告提出 QPEFT，把 LoRA-based adaptation、可学习量化参数、自适应激活量化和 SEEQ 结合起来，使模型在训练中逐步暴露于低比特反量化值。实验结果表明，QPEFT 在 ViT-B/16 和 ViT-L/16 的 3-bit/4-bit 设置下保持了有竞争力的分类适配性能，尤其在 ViT-L/16 4-bit 下接近 FP32 LoRA reference。

结论的核心不是“量化总是有效”，而是“量化与参数高效适配不应分开优化”。当 adapter 与 quantizer 在同一训练过程中协同工作时，低比特噪声对适配的破坏更容易被控制。

### 8.2 6.2 Future Work

未来工作包括：更完整的 multi-seed validation；更公平和更大规模的 PTQ/QAT/PEFT baseline；dense prediction tasks，如 detection 和 segmentation；对不同 ViT backbone、LoRA 注入位置和 quantizer 模块的 sensitivity analysis；引入 DoRA 等 PEFT 变体；真实 integer runtime 或 hardware-grounded deployment evaluation。

### 写作思路与 Review

Conclusion 没有新增结果，Future Work 方向合理。DoRA 被放在未来工作而不是当前结果来源，符合事实边界。需要在最终检查中确认 References 后无 Appendix 残留。

---

## 9. References 中文说明

References 使用 IEEE 数字编号 [1]–[31]，对应 Master Reference Registry。引用覆盖 Transformer、ViT、DeiT、Swin、VTAB、integer quantization、LSQ、Data-Free Quantization、AdaRound、BRECQ、QDrop、FQ-ViT、PTQ for ViT、PTQ4ViT、RepQ-ViT、AdaLog、I&S-ViT、APHQ-ViT、FIMA-Q、Q-ViT、OFQ、oscillation QAT、simulated quantization power savings、LoRA、Visual Prompt Tuning、AdaptFormer、Sparse-Tuning、DoRA、QLoRA 和 PEFT surveys。

Review 结论：当前引用系统是可用的。后续不要因为活跃项目文件中删除论文 PDF 就删除 registry 或 reference 条目。若新增文献，必须先进入 Pending Reference Queue。

---

## 10. 格式与行文 Review

### 10.1 已符合要求的部分

1. 主体结构符合 final report：Abstract、Acknowledgements、Contents、Introduction、Related Work、Materials and Methods、Results、Discussion、Conclusions and Future Work、References。
2. 当前无附录版本中，References 后不再保留模板 Appendix 和 plagiarism statement。
3. 正文中未发现明显中文残留或项目文件名残留。
4. 公式编号已经按全文顺序修复。
5. 引用编号使用全局 [1]–[31]。
6. 表格内容已压缩为聚合指标，避免 19 个数据集挤入正文。
7. Table captions 已比 v6 更接近论文风格。
8. Results 不再直接提 uploaded spreadsheets。

### 10.2 仍需最终处理的部分

1. Figure 1 需要插入最终高质量版本。
2. Figure 4 需要生成并插入。
3. Figure 5/6 需要从真实 TensorBoard 日志或 CSV 生成。
4. Figure 2/3 若无真实数据，应删除对应引用和图题。
5. 插图后必须更新 Contents、List of Figures、List of Tables。
6. 需要在 Word 中逐页检查表格是否跨页难看。
7. 需要最终导出 PDF 并检查页数是否在 30–40 页范围。
8. Acknowledgements 中 supervisor 可考虑在 hardcopy 前补全姓名，Moodle 提交则遵循模板留空要求。

### 10.3 是否仍存在 GPT 项目文件依赖式错误表达

根据当前 v7 检查版的文本检索和前后版本对比，没有发现以下高风险表达残留：

- 中文文件名；
- Excel 文件名；
- uploaded spreadsheets；
- GPT 项目文件；
- Codex；
- placeholder；
- planned；
- Appendix/appendices 指向；
- “result matrix” 作为内部文件来源的表达。

需要注意的是，正文中仍会合理出现 “evaluation results”、“diagnostics”、“case study”、“recorded training logs”等词，这些属于论文表达，不是项目内部文件泄漏。

### 10.4 是否过于保守或强调证据

当前版本的保守性总体合理。Introduction 和 Scope 明确 fake quantization、classification-oriented evaluation 和非硬件部署边界，符合工程 final report 的可信写法。Results 和 Discussion 中仍有部分表达较克制，例如 “diagnostic rather than statistical”，但这是为了避免单次 TensorBoard case study 被误解为多任务、多 seed 证明。整体不建议再大幅削弱边界说明。

若想进一步提升顶会风格，可把部分 “evidence” 词替换为更自然的学术表达，例如：

- evidence → observation / result / diagnostic record / empirical pattern；
- claim boundary → scope of inference / evaluation scope；
- not used as a substitute → does not replace；
- project setting → evaluated setting / present protocol。

---

## 11. 最终行动指南

### 11.1 立即处理

1. 插入 Figure 1。
2. 生成并插入 Figure 4。
3. 从真实 TensorBoard 或 CSV 生成 Figure 5/6。
4. 确认 Figure 2/3 是否有真实数据，无则删掉。
5. 更新目录、List of Figures、List of Tables。

### 11.2 最终检查

1. 全文搜索：中文字符、uploaded、spreadsheet、qpeft_exps、QPEFT_PTQ、placeholder、planned、reserved、Codex、Appendix。
2. 检查 Table 1–9 是否均先引用后出现。
3. 检查 Figure 1–6/7 是否均先引用后出现。
4. 检查公式编号和正文 Equation references 是否一致。
5. 检查 References 是否从新页开始。
6. 导出 PDF 逐页检查。

---

## 12. 总结

当前 v7 检查版已经基本完成正文从“写作稿”到“提交候选稿”的转换。它的主要优势是：研究主线明确，方法命名统一，引用体系稳定，结果表述从内部文件证据转向正式 evaluation protocol，claim boundary 清晰，且没有明显项目文件泄漏。后续最大风险不再是正文逻辑，而是图表质量、目录更新、表格分页和最终 PDF 视觉检查。只要 Figure 1、Figure 4、Figure 5、Figure 6 完成并通过最终格式审计，该报告可以进入提交前最终 polishing 阶段。
