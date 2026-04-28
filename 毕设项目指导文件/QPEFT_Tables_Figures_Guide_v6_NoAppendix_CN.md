# QPEFT Final Report 图表行动指南 v6（无附录版，中文重写版）

**项目题目：** Progressive Co-Optimization for Vision Transformer Quantization with Parameter-Efficient Fine-Tuning  
**适用版本：** 当前去掉附录后的 final report 工作版。Algorithm 已通过 LaTeX 方案解决，本指南不再展开 Algorithm。  
**核心目标：** 重新梳理当前正文已有表格与需要补齐的图，给出能直接喂给 GPT Image 2 和 Codex 的中英双语提示词，重点提升 Figure 1 和 Figure 4 的视觉质量、学术表达质量和可控性。

---

## 0. 总体原则

当前报告已经不适合继续把大量原始矩阵结果塞进正文。正文图表的任务是帮助读者快速理解研究动机、方法结构、实验协议和主要结论，而不是展示完整 19 个 VTAB-style 任务的所有逐项数字。

codex前置指令提示：针对我的毕业设计项目/TRS-SAS/linwei/QPEFT，参考QPEFT/毕设项目指导文件中的几份指导文件，和项目主目录下的word文档以及结果表，tensorboard结果图片和一组实验结果输出QPEFT/outputs_dev。

本版无附录情况下，应遵守以下原则：

1. 正文表格只展示 **Natural Avg.、Specialized Avg.、Structured Avg. 和 Tot. Avg.**，不展开 19 个数据集。
2. 正文表格采用接近 AI 顶级会议论文的三线表风格，但保持 Word 可编辑性。
3. Figure 1 是全文第一张方法总览图，应具有较高视觉吸引力，不能只是普通流程图。
4. Figure 4 是方法章节核心图，应比 Figure 1 更技术化，展示 QPEFT 如何作用于 ViT block。
5. Figure 5 和 Figure 6 必须来自真实 TensorBoard 日志或 CSV 导出，不允许用 AI 生成虚假曲线。
6. Figure 2 和 Figure 3 只有在有真实 sensitivity 数据或 activation dump 时才保留，否则应从正文删除。
7. 所有图应先在正文引用，再插入图像；图题放在图下方，且应能脱离正文独立理解。

---

## 1. 当前正文表格清单与处理建议

| 表格 | 位置 | 作用 | 当前处理建议 |
|---:|---|---|---|
| Table 1 | Section 1.2 | 对比 PTQ、QAT、PEFT 和 QPEFT 在低比特 ViT 适配中的定位。 | 保留。表题应为 1–2 行，只有 “Table 1.” 加粗，后续描述不加粗。 |
| Table 2 | Section 2.5 | 总结代表性文献流派与 QPEFT 研究缺口。 | 保留。用于让读者快速理解 Related Work 的收束逻辑。 |
| Table 3 | Section 3.7 | 总结分类导向 VTAB-style 适配与诊断实验设置。 | 保留。表题不能过短，不能只写 “Experimental setup.”，应明确覆盖内容。 |
| Table 4 | Section 3.8 | 说明报告可支持的结论范围与验证协议。 | 保留，但语气应像方法边界说明，不像内部证据审计表。 |
| Table 5 | Section 4.2 | ViT-B/16 和 ViT-L/16 的 QPEFT 主结果。 | 保留。仅展示三类任务 group averages 和 Tot. Avg.。 |
| Table 6 | Section 4.3 | 与 PTQ baselines 的对比。 | 保留但压缩，只保留代表性 matched baselines，不展示所有原始行。 |
| Table 7 | Section 4.4 | QAT-style baselines 与 full-image classification 补充结果。 | 保留。表题必须说明 full-image classification 是补充证据，不等同于 VTAB-style 19-task evidence。 |
| Table 8 | Section 4.5 | 消融实验摘要。 | 保留短表。不要把全部 schedule 行都塞进正文。必要时用 Figure 7 表达趋势。 |
| Table 9 | Section 4.7 | 可训练参数数量摘要。 | 保留。不要从参数数量直接推断真实硬件速度或能耗收益。 |

### 1.1 表格排版目标

正文表格建议采用 Word 中可实现的三线表近似风格：

1. 表题位于表格上方。
2. 表题中只有 **Table N.** 加粗，后续描述使用 Times New Roman 12 pt 常规字体。
3. 表题长度控制在 1–2 行；如果换行，仍使用 1.5 倍行距。
4. 表格正文使用 Times New Roman 12 pt，避免缩小到 7 pt 或 8 pt。
5. 表头行加粗。
6. 所有单元格水平居中、垂直居中，包括合并单元格。
7. 使用三线表风格：加粗上边线、表头下方细线、加粗下边线。
8. 除关键分组分隔外，不使用密集横线。
9. 不使用竖线。
10. 若表格跨页，优先通过压缩行数、合并表达或拆分为更短表格解决，而不是缩小字体。

---

## 2. 当前正文图规划

| 图 | 位置 | 状态 | 推荐生成路线 |
|---:|---|---|---|
| Figure 1 | Section 1.1 | 必须生成。全文第一张图，决定第一视觉印象。 | GPT Image 2 生成高质量矢量风格草图，再用 PowerPoint、Figma 或 draw.io 微调文字与布局。 |
| Figure 2 | Section 3.2 | 可选。只有真实 sensitivity 数据时保留。 | Codex + Python，从真实 layer/module sensitivity CSV 生成 heatmap。没有真实数据则删除。 |
| Figure 3 | Section 3.2 | 可选。只有真实 activation dump 或 histogram 数据时保留。 | Codex + Python，从真实张量导出或 histogram CSV 生成分布图。没有真实数据则删除。 |
| Figure 4 | Section 3.3 | 必须生成。方法章节核心图。 | 推荐先用 GPT Image 2 生成视觉草图，再用矢量工具或 Codex 生成可控版本。 |
| Figure 5 | Section 4.6 | 若有 TensorBoard scalar 导出则必须生成。 | Codex + Python/matplotlib，从 TensorBoard event 或 CSV 生成训练曲线。 |
| Figure 6 | Section 4.6 | 若有梯度日志则必须生成。 | Codex + Python/matplotlib，从 TensorBoard scalar 导出生成梯度诊断图。 |
| Figure 7 | Section 4.5 或 4.6 | 可选但推荐。 | Codex + Python/matplotlib，绘制 drop_epc 与 Tot. Avg. 的关系曲线，减轻 Table 8 压力。 |

---

## 3. 图片比例固定与去掉图例后被拉伸的问题

你观察到的现象是正常的：图像生成模型通常先在一个固定画布比例中构图，例如横向 16:9、1536 × 1024 或其他固定尺寸。模型并不是像 PowerPoint 那样“删除底部图例后保持原图不动”，而是会尝试重新填满整个画布。因此，当你只要求“去掉底部图例”时，模型可能会把主体三栏结构向下拉伸，以填补底部空出来的区域。这会导致图像变得瘦高、松散或不够顶会论文风格。

### 3.1 改进方法

生成图片时不要只写：

> remove the legend

而应同时写清楚：

1. 去掉底部图例。
2. 保持原有主体图的垂直比例。
3. 不要放大或纵向拉伸主体图。
4. 将主体图限制在中央内容边界内。
5. 允许底部和四周保留干净留白。
6. 三个面板保持相同列宽和稳定模块比例。

### 3.2 可直接加入所有 GPT Image 2 提示词的比例控制句

**英文：**

```text
Use a landscape canvas. Remove the bottom legend, but do not enlarge or vertically stretch the main diagram. Keep the three panels inside a compact central content bounding box, approximately 88% of the canvas width and 78% of the canvas height. Preserve equal column widths, stable module proportions, and clean white margins around the diagram.
```

**中文：**

```text
使用横向画布。去掉底部图例，但不要放大或纵向拉伸主体图。将三个面板限制在紧凑的中央内容边界内，约占画布宽度的 88% 和画布高度的 78%。保持三列等宽、模块比例稳定，并在图像四周保留干净留白。
```

### 3.3 插入 Word 时的比例控制

1. 插入图片后使用 “Lock aspect ratio”（锁定纵横比）。
2. 只调整宽度，不要单独调整高度。
3. 对复杂框架图，正文单栏 Word 页面中建议宽度约为 5.8–6.5 inches。
4. 如果图片过高，应回到图像生成或绘图工具中重新排版，而不是在 Word 中纵向压缩。

---

## 4. Figure 1：Introduction 高冲击力总览图

### 4.1 设计定位

Figure 1 是 Introduction 中的第一张图。它不只是说明 pipeline，更承担“第一视觉印象”的作用。它应当像 AI/CV 顶会论文中的 overview figure：结构清晰、数学符号准确、色彩高级、模块分层明确、没有多余装饰。

推荐图题：

> Figure 1. Conceptual overview of the proposed QPEFT workflow for low-bit Vision Transformer adaptation.

### 4.2 GPT Image 2 英文提示词

```text
Create a visually striking vector-style scientific schematic for the first figure of an AI/CV conference-quality research report. The figure should present a left-to-right three-stage pipeline for low-bit Vision Transformer adaptation.

Use a landscape canvas, preferably 1536 × 1024 or 16:9. Remove the bottom legend, but do not enlarge or vertically stretch the main diagram. Keep the three panels inside a compact central content bounding box, approximately 88% of the canvas width and 78% of the canvas height. Preserve equal column widths, stable module proportions, and clean white margins around the diagram.

Stage 1 title: “1 Pre-trained ViT backbone”. Show patch tokens, a class token, a compact token sequence formula, and three stacked Transformer blocks labeled ℓ = 1, ℓ = 2, and ℓ = L. Use a cool navy-blue palette. The panel should communicate frozen pretrained representation extraction.

Stage 2 title: “2 QPEFT wrapping”. Put a frozen “ViT backbone core” in the center. Around it place four clean modules: “LoRA adapters (rank-r)”, “Learnable quantization parameters”, “Adaptive activation quantization”, and “SEEQ stochastic exposure schedule”. Use teal for LoRA, amber for quantization, and violet for SEEQ. Use dashed control-flow arrows into the frozen ViT core. Include compact mathematical labels such as ΔW = BA, W_q = Q(W; s_w, z_w), X_q = Q(X; s_x, z_x), and p_t decreasing from 1 to 0.

Stage 3 title: “3 Low-bit adapted ViT”. Show quantized weights and activations entering a stack of Transformer blocks. Add a compact badge at the bottom saying “3-bit / 4-bit fake-quantized weights & activations”. Use small quantizer blocks labeled Q_w and Q_x on the left and right of the Transformer stack.

Design style: polished vector graphic, thin strokes, rounded rectangles, subtle shadows, crisp academic typography, no decorative icons, no photorealism, no clutter. Use a color system similar to modern CVPR/ICLR framework figures: navy for backbone, teal for LoRA, amber for quantization, violet for stochastic exposure, and a light gray-white background. All text must be horizontally aligned and readable. Do not invent additional modules, datasets, file names, hardware acceleration claims, or unrelated annotations.
```

### 4.3 GPT Image 2 中文提示词

```text
创建一张视觉效果强、矢量风格的科研示意图，作为 AI/CV 顶级会议质量研究报告的第一张图。图中展示低比特 Vision Transformer 适配的从左到右三阶段流程。

使用横向画布，优先为 1536 × 1024 或 16:9。去掉底部图例，但不要放大或纵向拉伸主体图。将三个面板限制在紧凑的中央内容边界内，约占画布宽度的 88% 和画布高度的 78%。保持三列等宽、模块比例稳定，并在图像四周保留干净留白。

第一阶段标题为 “1 Pre-trained ViT backbone”。展示 patch tokens、class token、简洁的 token sequence 公式，以及三个堆叠的 Transformer blocks，分别标注 ℓ = 1、ℓ = 2、ℓ = L。使用冷调深蓝色系。该面板应表达冻结的预训练表征提取过程。

第二阶段标题为 “2 QPEFT wrapping”。中间放置 frozen “ViT backbone core”。周围放置四个清晰模块：“LoRA adapters (rank-r)”、“Learnable quantization parameters”、“Adaptive activation quantization” 和 “SEEQ stochastic exposure schedule”。LoRA 使用青绿色，量化模块使用琥珀色，SEEQ 使用紫色。用虚线控制流箭头指向 frozen ViT core。加入紧凑数学标签，例如 ΔW = BA、W_q = Q(W; s_w, z_w)、X_q = Q(X; s_x, z_x)，以及从 1 逐渐下降到 0 的 p_t。

第三阶段标题为 “3 Low-bit adapted ViT”。展示 quantized weights 和 activations 进入 Transformer block stack。底部加入紧凑标识：“3-bit / 4-bit fake-quantized weights & activations”。在 Transformer stack 左右两侧加入小型量化器模块，分别标注 Q_w 和 Q_x。

设计风格：精致矢量图、细线条、圆角矩形、轻微阴影、清晰学术字体，不要装饰图标，不要照片风，不要拥挤。配色类似现代 CVPR/ICLR 方法图：backbone 用深蓝，LoRA 用青绿，量化用琥珀，随机暴露用紫色，背景为浅灰白。所有文字必须水平对齐且清晰可读。不要添加额外模块、数据集、文件名、硬件加速声明或无关注释。
```

### 4.4 人工后处理建议

GPT Image 2 生成后建议在 PowerPoint、Figma 或 draw.io 中检查并微调：

1. 所有英文标签是否拼写正确。
2. 数学符号是否符合正文公式。
3. 是否仍残留图例、无关图标或硬件暗示。
4. 三个 panel 的宽度和高度是否平衡。
5. 颜色是否过于饱和，是否影响打印可读性。

---

## 5. Figure 4：Methods 方法级 QPEFT 框架图

### 5.1 设计定位

Figure 4 不应重复 Figure 1 的三阶段总览。它应放大一个 ViT block，展示 QPEFT 在 block 内部如何插入 LoRA、量化器和 SEEQ gate。Figure 1 面向读者建立总体直觉，Figure 4 面向方法细节。

推荐图题：

> Figure 4. Method-level QPEFT framework inside a Vision Transformer block.

### 5.2 GPT Image 2 英文提示词

```text
Create a method-level framework figure for a research report on quantized parameter-efficient fine-tuning of Vision Transformers. This figure should zoom into one Vision Transformer block and show how QPEFT modifies the block. It should not repeat the high-level three-stage overview.

Use a wide landscape canvas. Keep the diagram compact and do not stretch modules vertically. Use a clean central computational path with side branches. Use four color-coded components: frozen ViT computation in navy blue, trainable LoRA updates in teal, quantizers in amber, and SEEQ stochastic exposure in violet.

Main structure: input tokens pass through LayerNorm, multi-head self-attention, residual connection, MLP with GELU, and a second residual connection. Show qkv, proj, fc1, and fc2 as target linear layers. Add small LoRA branches beside qkv, proj, fc1, and fc2 with the label ΔW = BA. Add weight quantizer boxes and activation quantizer boxes. Mark post-softmax attention and post-GELU activation as special quantization points. Show an SEEQ gate that selects between F_i and Q_i element-wise and outputs O_i, with probability controlled by r_t.

Use exact labels only: “Frozen ViT path”, “LoRA update”, “Weight quantizer”, “Activation quantizer”, “post-softmax”, “post-GELU”, “SEEQ gate”, “F_i”, “Q_i”, “O_i”, “r_t”. Do not include datasets, file names, hardware speedup, unrelated icons, or evaluation results.

Design style: polished CVPR/ICLR framework figure, precise layout, thin arrows, rounded blocks, subtle shadows, high readability, restrained colors, and clear mathematical labels. Avoid clutter. Keep enough white space between modules.
```

### 5.3 GPT Image 2 中文提示词

```text
创建一张用于研究报告的方法级框架图，主题是 Vision Transformer 的量化参数高效微调。该图应放大一个 Vision Transformer block，展示 QPEFT 如何改造该 block。不要重复 Figure 1 的三阶段总览图。

使用宽幅横向画布。主体图保持紧凑，不要纵向拉长模块。使用清晰的中央计算路径和侧边分支。使用四种颜色编码组件：frozen ViT computation 用深蓝，trainable LoRA updates 用青绿色，quantizers 用琥珀色，SEEQ stochastic exposure 用紫色。

主体结构：input tokens 依次经过 LayerNorm、multi-head self-attention、residual connection、带 GELU 的 MLP，以及第二个 residual connection。展示 qkv、proj、fc1、fc2 作为目标线性层。在 qkv、proj、fc1、fc2 旁边加入小型 LoRA 分支，并标注 ΔW = BA。加入 weight quantizer boxes 和 activation quantizer boxes。将 post-softmax attention 和 post-GELU activation 标记为特殊量化点。展示一个 SEEQ gate，该门控在 F_i 和 Q_i 之间逐元素选择，并在 r_t 控制的概率下输出 O_i。

只使用以下准确标签：“Frozen ViT path”、“LoRA update”、“Weight quantizer”、“Activation quantizer”、“post-softmax”、“post-GELU”、“SEEQ gate”、“F_i”、“Q_i”、“O_i”、“r_t”。不要包含数据集、文件名、硬件加速、无关图标或实验结果。

设计风格：精致的 CVPR/ICLR 方法图，布局准确，细箭头，圆角模块，轻微阴影，高可读性，配色克制，数学标签清晰。避免拥挤，模块之间保留足够留白。
```

---

## 6. Figure 2：Layerwise Sensitivity Heatmap

### 6.1 使用条件

只有存在真实 sensitivity CSV 时才生成。可用指标包括 quantization error、gradient norm、validation drop 或其他明确定义的 sensitivity metric。没有真实数据时，不应在正文保留此图。

### 6.2 Codex 英文提示词

```text
Write a Python script that reads a CSV file containing layer index, module name, and sensitivity value for a Vision Transformer. The module names may include qkv, proj, fc1, fc2, post-GELU, and post-softmax. Generate a publication-quality heatmap with Transformer block index on the y-axis and module type on the x-axis. Use matplotlib only, no seaborn. Use a restrained academic color palette, clear tick labels, and a colorbar labeled “Sensitivity”. Export SVG, PDF, and 600 dpi PNG. The figure must not invent any values and must fail with a clear error if the CSV file is missing required columns.
```

### 6.3 Codex 中文完整对照

```text
针对我的毕业设计项目/TRS-SAS/linwei/QPEFT，参考QPEFT/毕设项目指导文件中的几份指导文件，和项目主目录下的word文档以及结果表，tensorboard结果图片和一组实验结果输出QPEFT/outputs_dev。
请编写一个 Python 脚本，读取一个 CSV 文件。该 CSV 文件包含 Vision Transformer 的 layer index、module name 和 sensitivity value 三类信息。module name 可以包括 qkv、proj、fc1、fc2、post-GELU 和 post-softmax。脚本需要生成一张适合AI顶级会议论文发表的 heatmap，其中 y 轴为 Transformer block index，x 轴为 module type，颜色表示 sensitivity value。只使用 matplotlib，不要使用 seaborn。配色应克制并具有学术风格，坐标轴刻度清晰，颜色条标注为 “Sensitivity”。导出 SVG、PDF 和 600 dpi PNG 三种格式。该图不能凭空生成任何数值。如果 CSV 缺少必要列，脚本必须给出清晰错误提示并停止运行。
```

针对我的毕业设计项目/TRS-SAS/linwei/QPEFT，参考QPEFT/毕设项目指导文件中的几份指导文件，和项目主目录下的word文档以及结果表，tensorboard结果图片和一组实验结果输出QPEFT/outputs_dev，生成/TRS-SAS/linwei/QPEFT/毕设项目指导文件/QPEFT_Tables_Figures_Guide_v6_NoAppendix_CN.md要求的Figure 2，如

---

## 7. Figure 3：Activation Distribution Analysis

### 7.1 使用条件

只有存在真实 activation dump、tensor dump 或导出的 histogram CSV 时才生成。该图可用于比较普通 activation、post-GELU activation 和 post-softmax attention probabilities 的分布差异。

### 7.2 Codex 英文提示词

```text
Write a Python script that reads activation dump files or exported histogram CSV files for ordinary activations, post-GELU activations, and post-softmax attention probabilities in a Vision Transformer. Generate a clean academic distribution figure comparing the three tensor types. Use matplotlib only, no seaborn. Prefer separate panels if scales differ greatly. Label axes clearly and export SVG, PDF, and 600 dpi PNG. The script must not synthesize fake distributions unless a separate --demo flag is explicitly set, and demo mode must never be used for the final report.
```

### 7.3 Codex 中文完整对照

```text
请编写一个 Python 脚本，读取 Vision Transformer 中普通 activations、post-GELU activations 和 post-softmax attention probabilities 的 activation dump 文件或已导出的 histogram CSV 文件。生成一张干净、学术风格的分布对比图，用于比较这三类张量的分布特征。只使用 matplotlib，不要使用 seaborn。如果三类数据的尺度差异很大，优先使用多个 panel 分开展示，而不是强行放在同一个坐标轴上。坐标轴标签必须清晰，并导出 SVG、PDF 和 600 dpi PNG 三种格式。脚本不能合成虚假的分布数据，除非用户显式设置单独的 --demo 参数；demo 模式绝不能用于最终报告。
```

---

## 8. Figure 5：CIFAR Training Trajectory

### 8.1 使用条件

Figure 5 应从 TensorBoard event 文件或导出的 scalar CSV 生成。它用于展示 CIFAR 3-bit QPEFT case study 的训练过程，不应被写成多任务收敛证明。

### 8.2 Codex 英文提示词

```text
Write a Python script that reads TensorBoard event files or exported scalar CSV files for the CIFAR 3-bit QPEFT case study. Plot validation Acc@1 against epoch. Use matplotlib only, no seaborn. Use a clean academic line plot with readable axis labels, modest marker size, and no heavy chart border. Annotate the best validation accuracy and the final epoch if available. Export SVG, PDF, and 600 dpi PNG. The figure title should not claim multi-task convergence. Use the caption wording “CIFAR case-study training trajectory for 3-bit QPEFT.”
```

### 8.3 Codex 中文完整对照

```text
请编写一个 Python 脚本，读取 CIFAR 3-bit QPEFT case study 的 TensorBoard event 文件或已导出的 scalar CSV 文件。绘制 validation Acc@1 随 epoch 变化的曲线。只使用 matplotlib，不要使用 seaborn。图像应采用干净的学术折线图风格，坐标轴标签清晰，marker 大小适中，不要使用很重的图框。若数据中包含 best validation accuracy 和 final epoch，请在图中标注。导出 SVG、PDF 和 600 dpi PNG 三种格式。图标题不能声称多任务收敛。建议 caption 使用 “CIFAR case-study training trajectory for 3-bit QPEFT.”
```

---

## 9. Figure 6：Gradient Diagnostics

### 9.1 使用条件

Figure 6 应从 TensorBoard 梯度日志或导出的 scalar CSV 生成。它是训练动力学个案诊断，不是广泛统计验证。

### 9.2 Codex 英文提示词

```text
Write a Python script that reads TensorBoard event files or exported scalar CSV files for representative LoRA and quantizer-related gradient diagnostics. Plot selected traces such as Grad_qkv.lora_B, Grad_mlp.fc2.lora_B, and Grad_attn_quantizer.log_base_alpha for representative blocks. Use matplotlib only, no seaborn. If magnitudes differ strongly, use separate panels rather than forcing all traces onto one axis. Export SVG, PDF, and 600 dpi PNG. The caption must state that the figure is a CIFAR case-study diagnostic rather than broad statistical validation.
```

### 9.3 Codex 中文完整对照

```text
请编写一个 Python 脚本，读取代表性 LoRA 参数和 quantizer-related 参数的 TensorBoard event 文件或已导出的 scalar CSV 文件，用于生成 gradient diagnostics 图。绘制若干代表性曲线，例如 Grad_qkv.lora_B、Grad_mlp.fc2.lora_B，以及代表性 block 中的 Grad_attn_quantizer.log_base_alpha。只使用 matplotlib，不要使用 seaborn。如果不同曲线的数量级差异很大，应使用分面板展示，而不是强行放在同一个 y 轴中。导出 SVG、PDF 和 600 dpi PNG 三种格式。caption 必须说明该图是 CIFAR case-study diagnostic，而不是广泛统计验证。
```

---

## 10. Optional Figure 7：drop_epc Schedule Curve

### 10.1 使用条件

如果正文 Table 8 因 drop_epc 消融而显得拥挤，可以用 Figure 7 表示 drop_epc 与 Tot. Avg. 的关系。完整 schedule 行不必全部保留在正文表格中。

### 10.2 Codex 英文提示词

```text
Write a Python script that reads the ablation table for drop_epc values and total average accuracy. Plot drop_epc on the x-axis and Tot. Avg on the y-axis. Highlight the best setting and shade or annotate the stable range around drop_epc 04 to 07 if present in the data. Use matplotlib only, no seaborn. Export SVG, PDF, and 600 dpi PNG. The plot should support the ablation discussion but must not imply that the schedule is theoretically optimal beyond the evaluated settings.
```

### 10.3 Codex 中文完整对照

```text
请编写一个 Python 脚本，读取包含 drop_epc 取值和 total average accuracy 的消融实验表。绘制 drop_epc 与 Tot. Avg. 的关系曲线，其中 x 轴为 drop_epc，y 轴为 Tot. Avg.。如果数据中存在最佳设置，请高亮该点；如果 drop_epc 04 到 07 附近存在稳定区间，请用阴影或注释标出。只使用 matplotlib，不要使用 seaborn。导出 SVG、PDF 和 600 dpi PNG 三种格式。该图应服务于消融实验讨论，但不能暗示该 schedule 在已评估设置之外也具有理论最优性。
```

---

## 11. 推荐插图流程

1. Figure 1 和 Figure 4 可以先用 GPT Image 2 生成高质量草图。
2. 对 GPT Image 2 生成图，应人工检查文字、数学符号和模块名称。
3. 如果图中文字错误较多，建议在 PowerPoint、Figma 或 draw.io 中重画文字层。
4. Figure 2、Figure 3、Figure 5、Figure 6 和 Figure 7 必须用真实数据通过 Python 生成。
5. 每张图建议同时导出 SVG、PDF 和 600 dpi PNG。
6. 插入 Word 时优先尝试 SVG。
7. 如果 SVG 字体或数学符号错位，则使用 600 dpi PNG。
8. Word 中图片设置为 “In Line with Text”，并居中。
9. 只按宽度缩放图片，保持纵横比锁定。
10. 图题放在图下方，表题放在表上方。
11. 每张图必须在正文中先被引用，再出现。

---

## 12. 当前最优行动顺序

1. 用新版 Figure 1 提示词重新生成总览图，优先使用无底部图例、比例稳定、视觉冲击力更强的版本。
2. 用 Figure 4 提示词生成方法级框架图草图，再人工或用矢量工具清理。
3. 判断 Figure 2 和 Figure 3 是否有真实数据。如果没有，直接从正文删除，不要保留 placeholder 或 planned language。
4. 用 Codex 生成 Figure 5 和 Figure 6 的 Python 绘图脚本，从真实 TensorBoard logs 或 CSV 导出数据作图。
5. 如果 Table 8 仍然拥挤，用 Codex 生成 Figure 7，并将正文 Table 8 压缩为 summary table。
6. 插入所有图后，在 Word 中更新 Contents、List of Figures 和 List of Tables。
7. 导出 PDF，逐页检查图表是否清晰、是否跨页异常、是否存在未更新编号。

---

## 13. 本版相对上一版的主要改动

1. 全文改为中文说明，不再出现大段英文 guide text。
2. GPT Image 2 提示词保留中英双语，便于直接使用英文 prompt 生成，同时保留中文审核版本。
3. 所有 Codex 提示词均补充完整中文对照。
4. Figure 1 提示词显著增强视觉标准，参考 AI/CV 顶级会议 overview figure 风格。
5. 明确解释“去掉底部图例后图像被拉长”的原因，并把比例控制句融入 Figure 1 和 Figure 4 的提示词策略中。
6. 明确无附录版本下正文图表的取舍原则，避免再出现 Appendix 指向。
