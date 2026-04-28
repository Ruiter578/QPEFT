# 毕设提纲与 AI 工作流 v6：定海神针版

**Project:** Progressive Co-Optimization for Vision Transformer Quantization with Parameter-Efficient Fine-Tuning  
**Current controlling report:** `Final_Report_QPEFT_revised_v7_tables_polished检查版.docx`  
**Current reference authority:** `Master_Reference_Registry_v5_Current_Aligned.md`  
**Current figure/table authority:** `QPEFT_Tables_Figures_Guide_v6_NoAppendix_CN.md`  
**Version role:** v6 是当前项目后续工作的唯一总控指导文件。它吸收 v4 的总体毕业设计结构、v5 的章节生成流程和结果证据规划，同时根据 v7 检查版全文、当前引用注册表、无附录图表指南和已插入 LaTeX Algorithm 的事实，清理历史版本冲突，作为后续图表生成、全文微调和最终审计的核心依据。

---

## 0. v6 的核心结论

1. 当前项目已经从“正文生成阶段”进入“最终装配与审计阶段”。Introduction、Related Work、Materials and Methods、Results、Discussion、Conclusions and Future Work、References 的主体内容已经形成。后续不应再大幅重写正文，除非发现事实错误、格式错误或图表引用不一致。
2. 当前最终主线是 **Quantized Parameter-Efficient Fine-Tuning (QPEFT，量化参数高效微调)**，技术实现叙事为 **LoRA-based QPEFT**，不是 DiLoRA-based QPEFT，也不是 DoRA-based QPEFT。
3. SEEQ 的最终命名为 **Stochastic Element-wise Exposure to Quantization（随机逐元素量化暴露）**。它用于描述训练时在 full-precision values（全精度值）和 dequantized low-bit values（低比特反量化值）之间进行随机逐元素暴露的机制。代码层面可用 DropSTE-style stochastic exposure 解释。
4. 当前 v7 检查版已经删除 References 之后的附录与 plagiarism statement 模板残留，因此后续正文不要写 “Appendix” 或 “appendices”。如果最后重新决定加入附录，必须同步恢复目录、正文指向和附录编号。
5. 当前引用体系以 `Master_Reference_Registry_v5_Current_Aligned.md` 为准，稳定维护 [1]–[31]。当前正文应保持每条引用均至少出现一次，不再每章重新编号。
6. 当前图表体系以 `QPEFT_Tables_Figures_Guide_v6_NoAppendix_CN.md` 为准。正文表格保留 Table 1–Table 9，正文不展示 19 个 VTAB-style 数据集逐项列，只展示 Natural Avg.、Specialized Avg.、Structured Avg. 和 Tot. Avg. 等聚合指标。
7. Algorithm 已通过 LaTeX 生成裁剪图片方案解决，后续不再作为主要工作流问题。只需要检查其 Word 插入位置、清晰度、标题编号和正文引用是否一致。
8. 当前剩余最高优先级是图像生成和插入：Figure 1、Figure 4、Figure 5、Figure 6 必须重点完成；Figure 2/3 只有真实 sensitivity 或 activation dump 数据时才保留；Figure 7 可选，用于展示 drop_epc 趋势。

---

## 1. 活跃项目文件与归档策略

### 1.1 活跃文件

后续对话和修改应优先读取以下文件：

| 文件 | 角色 | 使用方式 |
|---|---|---|
| `Final_Report_QPEFT_revised_v7_tables_polished检查版.docx` | 当前全文主版本 | 所有后续图表插入、最终格式检查和正文微调以此为准。 |
| `Master_Reference_Registry_v5_Current_Aligned.md` | 引用体系凭证 | 维护 [1]–[31]、RefKey、IEEE 参考文献、引用位置和文献用途。 |
| `QPEFT_Tables_Figures_Guide_v6_NoAppendix_CN.md` | 图表行动指南 | 指导 Figure 1–7 的生成、正文表格取舍、无附录版本图表边界。 |
| `毕设提纲与AI工作流 v5最新版.pdf` | 历史执行规范与结果证据规划 | 作为 v6 的上游文件；之后可只在追溯时读取。 |
| `AI+代码分析（最新版）.pdf` | 代码事实控制文件 | 确认 LoRA、DropSTE、量化器、训练流程和参数设置，避免方法表述漂移。 |
| `qpeft_exps.xlsx` | QPEFT 主结果与 QAT/full-image/ablation 数据源 | 只用于数据核验和图表生成，不应在正式正文中出现文件名。 |
| `QPEFT_PTQ对照实验结果.xlsx` | PTQ baseline 对照数据源 | 只用于数据核验和 Table 6 生成，不应在正式正文中出现文件名。 |
| `Instructions on Project reports.pdf` | 官方格式要求 | 约束页数、图表标题、引用、公式、附录与图表位置。 |
| `Final Report Template.docx` | 学校模板 | 只用于格式参考，不再复制模板残留内容。 |
| `PTQ-Adalog.pdf` | 风格参考与核心 PTQ 文献 | 可参考图表风格和 AdaLog 方法背景。 |

### 1.2 可归档文件

以下文件可移动到 `Archive/` 或从活跃项目文件区移除（已执行）

1. 旧版 v4/v5 指导文件，只保留一份 v6 作为后续主控。
2. 旧版 Project Factsheet v4。若保留，应标注 “superseded”。
3. 旧版全文 review、旧版中文翻译和旧版表格指南，因为它们已被 v7 和 v6 guide 覆盖。
4. 已经不再需要直接读取的论文 PDF。大部分论文 PDF 可移出活跃区，但不建议永久删除。Registry 已能提供引用映射，但答辩前仍可能需要回查原文。

### 1.3 文件命名建议

后续最终文件建议命名为：

- `Final_Report_QPEFT_v8_figures_inserted.docx`
- `Final_Report_QPEFT_v9_final_audited.docx`
- `Final_Report_QPEFT_Final_Submission.docx`
- `Final_Report_QPEFT_Final_Submission.pdf`

原则：不要同时保留多个“final new”或“final revised”名称，避免版本歧义。

---

## 2. 方法命名与事实边界

### 2.1 最终方法命名

| 名称 | 全称 | 中文 | 最终用途 |
|---|---|---|---|
| QPEFT | Quantized Parameter-Efficient Fine-Tuning | 量化参数高效微调 | 总框架名，贯穿全文。 |
| SEEQ | Stochastic Element-wise Exposure to Quantization | 随机逐元素量化暴露 | 方法组件名，解释渐进量化暴露。 |
| LoRA | Low-Rank Adaptation | 低秩适应 | 当前 PEFT 主实现。 |
| PTQ | Post-Training Quantization | 后训练量化 | Related Work 和 baseline 对比。 |
| QAT | Quantization-Aware Training | 量化感知训练 | Related Work 和 QAT-style baseline 背景。 |
| DropSTE | Drop Straight-Through Estimator | 随机保留式直通估计器 | 代码层面的随机暴露/rounding 解释。 |
| DoRA | Weight-Decomposed Low-Rank Adaptation | 权重分解低秩适应 | 仅可作为 future work 或扩展接口，不作为当前结果来源。 |
| DiLoRA | Disentangled Low-Rank Adaptation | 解耦低秩适应 | 不作为最终主方法。 |

### 2.2 推荐方法定位句

正式正文中可使用：

> The proposed method is best described as quantization-aware parameter-efficient fine-tuning, or as a fake-QAT-style QPEFT framework for Vision Transformer adaptation.

中文理解：本项目最准确的定位是“量化感知参数高效微调”，或“面向 Vision Transformer 适配的伪 QAT 风格 QPEFT 框架”。

### 2.3 禁止或延后声明

不得写：

1. QPEFT is a PTQ method.
2. QPEFT is a standard QAT method.
3. QPEFT fully replaces PTQ and QAT.
4. QPEFT achieves state-of-the-art performance.
5. QPEFT proves universal superiority over all PTQ/QAT baselines.
6. QPEFT achieves measured real hardware acceleration.
7. The method is deployment-ready on edge hardware.
8. The method has been validated on dense prediction tasks.
9. DiLoRA is the implemented main method.
10. DoRA is responsible for the reported result.
11. Figure 2/3 shows real sensitivity or activation distributions unless true data exist.
12. Appendix contains the complete results, unless appendices are reintroduced.

---

## 3. 当前全文结构与提纲

当前 v7 检查版采用以下结构：

1. Abstract
2. Keywords
3. Acknowledgements
4. Contents
5. List of Figures
6. List of Tables
7. 1 Introduction
8. 2 Related Work
9. 3 Materials and Methods
10. 4 Results
11. 5 Discussion
12. 6 Conclusions and Future Work
13. References

当前不使用 Appendix。

> list of figures与list of tables放在后期处理。

### 3.1 Introduction

目标：建立研究动机、问题、挑战、方法概览和 claim boundary。  
核心图表：Figure 1、Table 1。  
关键审查点：

- 不出现内部文件名、上传文件名、项目文件名。
- 不把 QPEFT 描述为硬件部署方案。
- 不将 LoRA/PEFT 与真实推理加速直接等同。
- Figure 1 必须是高质量总览图，不应只是普通流程图。

### 3.2 Related Work

目标：按 ViT、PTQ、QAT、PEFT 和 research gap 组织文献，不做流水账。  
核心图表：Table 2。  
关键审查点：

- 每个引用必须可映射到 registry。
- PTQ、QAT、PEFT 的局限必须服务于 QPEFT 的研究空缺。
- QLoRA 只能作为相关量化 PEFT 背景，不可等同于本项目。

### 3.3 Materials and Methods

目标：形式化 low-bit fake quantization、LoRA、QPEFT objective、SEEQ、量化模块、训练协议、实验设置和 claim scope。  
核心图表：Figure 2/3 可选，Figure 4 必须，Table 3 和 Table 4 保留。  
关键审查点：

- 公式必须用 Word 原生公式对象或已正确插入。
- 公式编号全文连续。
- 3.7 Experimental Setup 应像论文方法章节，而不是内部数据清单。
- 3.8 Claim Scope and Validation Protocol 应是技术边界说明，不是 GPT 审计表。

### 3.4 Results

目标：报告主结果、PTQ 对照、QAT/full-image 补充结果、ablation、TensorBoard case study 和参数效率。  
核心表格：Table 5–Table 9。  
核心图：Figure 5、Figure 6，Figure 7 可选。  
关键审查点：

- 不出现 `qpeft_exps.xlsx`、`QPEFT_PTQ...xlsx`、uploaded、spreadsheet 等正文不可接受信息。
- 主结果使用 Tot. Avg.，同时提供 Natural/Specialized/Structured group averages。
- Table 6 只说 matched observations，不说 universal SOTA。
- Table 8 不把 shift/drop_epc 写成已完全理论化模块。

### 3.5 Discussion

目标：解释结果背后的机制、边界和局限。  
关键审查点：

- 可以解释 model scale effect、bit-width sensitivity、PTQ/QAT/PEFT relation、ablation implication 和 TensorBoard diagnostics。
- 不能新增 Results 没有报告的数据。
- 避免过度保守到像内部审计，也避免过度夸大到像营销 claim。

### 3.6 Conclusions and Future Work

目标：总结项目贡献、结果和未来工作。  
关键审查点：

- 不引入新实验。
- Future Work 可以用 bullet points。
- DoRA、multi-seed、dense prediction、hardware-grounded evaluation 等只能作为未来方向。

---

## 4. Word 格式硬约束

| 元素 | 约束 |
|---|---|
| 一级标题 | Times New Roman, 16 pt, bold, black。 |
| 二级标题 | Times New Roman, 14 pt, bold, black。 |
| 三级标题 | Times New Roman, 12 pt, bold, black。 |
| 正文 | Times New Roman, 12 pt, 1.5 line spacing, justified。 |
| 正文间距 | 段前 3 pt，段后 0。 |
| 缩进 | 每个 subsection 标题后第一段不缩进；后续正文首行缩进。 |
| 图题 | 图下方，stand-alone caption，先正文引用后插图。 |
| 表题 | 表上方，stand-alone caption，先正文引用后插表。 |
| 表格 | Word 可编辑三线表，表头加粗，所有单元格水平和垂直居中。 |
| 公式 | Word 原生公式或高质量矢量图；编号在公式所在行右侧，全文连续。 |
| 引用 | IEEE 数字编号 [n]，全局稳定，References 另起新页。 |
| 页数 | BEng 正文不少于 30 页，不超过 40 页；图不超过 20 张。 |

---

## 5. 当前图表规划

### 5.1 表格

| 表格 | 当前策略 |
|---:|---|
| Table 1 | 保留。PTQ/QAT/PEFT/QPEFT 定位表。 |
| Table 2 | 保留。Representative literature map。 |
| Table 3 | 保留。Experimental setup for classification-oriented VTAB-style adaptation and diagnostic evaluation。 |
| Table 4 | 保留。Claim scope and validation protocol。 |
| Table 5 | 保留。Main QPEFT results，ViT-B/16 和 ViT-L/16。 |
| Table 6 | 保留。代表性 PTQ baselines，对照 AdaLog/QDrop/RepQ-ViT/PTQ4ViT 的必要行。 |
| Table 7 | 保留。QAT-style baselines 与 full-image classification 补充结果。 |
| Table 8 | 保留短表。Shift 与 stochastic exposure schedule 摘要，不展示全部 long schedule。 |
| Table 9 | 保留。Trainable parameter summary。 |

### 5.2 图

| 图 | 当前策略 |
|---:|---|
| Figure 1 | 必须生成并插入。Introduction 的第一视觉锚点，应高质量、类似 CVPR/ICLR overview figure。 |
| Figure 2 | 可选。 |
| Figure 3 | 可选。 |
| Figure 4 | 必须生成并插入。Methods 核心图，应展示 ViT block 内部 QPEFT/LoRA/quantizer/SEEQ 作用路径。 |
| Figure 5 | 建议生成。必须来自 TensorBoard scalar 或 CSV，不允许 AI 伪造曲线。 |
| Figure 6 | 建议生成。必须来自 TensorBoard gradient logs 或 CSV，不允许 AI 伪造曲线。 |
| Figure 7 | 可选。用真实 ablation 数据绘制 drop_epc 与 Tot. Avg. 趋势。 |

---

## 6. 当前结果事实

### 6.1 QPEFT 主结果

| Backbone | Setting | W/A | Natural Avg | Specialized Avg | Structured Avg | Tot. Avg |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 | FP32 LoRA | 32/32 | 81.34 | 86.14 | 60.30 | 73.49 |
| ViT-B/16 | QPEFT | 3/3 | 74.18 | 84.59 | 58.92 | 69.95 |
| ViT-B/16 | QPEFT | 4/4 | 75.78 | 85.37 | 60.14 | 71.21 |
| ViT-L/16 | FP32 LoRA | 32/32 | 83.35 | 86.53 | 58.87 | 73.71 |
| ViT-L/16 | QPEFT | 2/2 | 68.34 | 79.20 | 42.22 | 61.92 |
| ViT-L/16 | QPEFT | 3/3 | 80.16 | 85.27 | 60.58 | 72.99 |
| ViT-L/16 | QPEFT | 4/4 | 81.81 | 86.08 | 60.36 | 73.68 |

### 6.2 PTQ 对照关键结论

- ViT-B/16 3-bit：QPEFT Tot. Avg. 69.95；QDrop 3-bit 为 61.72；AdaLog 3-bit 为 60.35。
- ViT-B/16 4-bit：QPEFT 71.21；AdaLog 70.16；QDrop 67.37。
- ViT-L/16 3-bit：QPEFT 72.99；AdaLog 48.05。
- ViT-L/16 4-bit：QPEFT 73.68；AdaLog 69.83。

写法：可以说 “competitive” 或 “stronger in matched reported settings”，不可说 “universal SOTA”。

### 6.3 QAT-style baseline 与 full-image classification

- QAT-style baseline: 3-bit Tot. Avg. 52.79；4-bit Tot. Avg. 57.83。
- Full-image classification 是补充结果，不等同于 VTAB-style 19-task 主结果。

### 6.4 Ablation

- no-shift Tot. Avg. 64.56。
- post-GELU constant shift 0.17 Tot. Avg. 69.84。
- drop_epc=05 Tot. Avg. 70.08。
- drop_epc=04 到 07 保持约 70.0 稳定区间。

写法：只能写成 empirical ablation factor，不写成严格证明的理论模块。

---

## 7. 后续 AI 工作流 v6

### Stage A：图表最终生成

输入物：v7 全文、图表 guide、结果表、TensorBoard 日志、算法图片。  
输出物：Figure 1、Figure 4、Figure 5、Figure 6，可选 Figure 7。

检查：

- 图片中无中文、无文件名、无项目内部术语。
- Figure 1 是 overview，不塞入过多 ViT block 内部细节。
- Figure 4 是 mechanism，不与 Figure 1 重复。
- Figure 5/6 来自真实日志。
- 图像插入 Word 后保持纵横比，不拉伸。

### Stage B：图表插入与交叉引用检查

输入物：v7 全文和生成好的图像。  
输出物：v8 figures inserted 版。

检查：

- 每张图首次引用后插入。
- 图题在图下方。
- 表题在表上方。
- Contents、List of Figures、List of Tables 更新。
- 图表编号连续。

### Stage C：最终正文语言审计

输入物：v8 全文。  
输出物：v9 language-audited 版。

重点查：

- 中文字符。
- 项目文件名。
- uploaded、spreadsheet、placeholder、planned、reserved、Codex、Appendix 等不应出现词。
- 过度保守表达，例如反复说 “evidence source” “uploaded result matrix”。
- 过度夸大表达，例如 SOTA、first、hardware acceleration。

### Stage D：最终格式审计

输入物：v9 全文。  
输出物：final-audited DOCX + PDF。

检查：

- 页数 30–40。
- 图不超过 20 张。
- 标题格式、正文格式、表格格式。
- 公式编号连续。
- References 另起新页。
- 封面、Abstract、Acknowledgements、Contents 完整。

### Stage E：最终提交前人工审查

人工必须逐页看 PDF：

1. 封面信息是否准确。
2. Supervisor 是否按 Moodle/hardcopy 要求留空或填写。
3. Abstract 是否约 100–250 words 的学校要求与项目实际之间可接受。
4. 图表是否清晰。
5. Table 5–9 是否跨页难看。
6. References 是否完整。
7. 页码和目录是否正确。

---

## 8. 最终审计清单

### 8.1 Content Audit

- [ ] QPEFT、SEEQ、LoRA、DropSTE-style exposure 命名一致。
- [ ] DiLoRA 不作为主方法。
- [ ] DoRA 只作为 future work。
- [ ] PTQ/QAT/PEFT/QPEFT 定位准确。
- [ ] Results 不新增未验证结论。
- [ ] Discussion 不新增未报告结果。
- [ ] Conclusions 不夸大。

### 8.2 Evidence Audit

- [ ] Table 5 数字与结果表一致。
- [ ] Table 6 数字与已修正 PTQ 对照一致。
- [ ] Table 7 明确 full-image classification 是补充结果。
- [ ] Table 8 不过度解释 shift/drop_epc。
- [ ] Table 9 不推断真实硬件收益。

### 8.3 Reference Audit

- [ ] 正文每个 [n] 均存在于 References。
- [ ] References 每一条均被正文引用。
- [ ] [1]–[31] 与 registry 一致。
- [ ] 新增文献先进 Pending Queue。
- [ ] References 格式为 IEEE。

### 8.4 Figure/Table Audit

- [ ] Figure 1 已插入并首次引用。
- [ ] Figure 4 已插入并首次引用。
- [ ] Figure 5/6 若保留，必须来自真实 TensorBoard 数据。
- [ ] Figure 2/3 尽量使用真实数据或者代码、命令得到的结果。
- [ ] 所有图题在图下。
- [ ] 所有表题在表上。
- [ ] 表格三线表风格统一。

### 8.5 Language Audit

- [ ] 无中文字符。
- [ ] 无项目文件名。
- [ ] 无 uploaded / spreadsheet / placeholder / planned / reserved / Codex / Appendix 残留。
- [ ] 不使用 “state-of-the-art” 除非严格限定。
- [ ] 不使用 “first” 之类不可证贡献。
- [ ] 不使用大段内部审计式语言。

---

## 9. v6 后续使用规则

从现在开始，后续所有生成和修改都应先默认使用本 v6 文件作为最高级项目指导文件。若本文件与旧版 v4/v5 冲突，以 v6 为准；若 v6 与官方 Instructions 冲突，以官方 Instructions 为准；若 v6 与当前 v7 全文冲突，应先判断是正文过时还是 v6 过时，再明确更新来源。不要再同时引用多个旧版指导文件作为等权控制文件。
