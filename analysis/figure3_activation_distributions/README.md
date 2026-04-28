# Figure 3 Activation Distribution Analysis

This directory contains the reproducible export and plotting code for Figure 3.

## Data source

The generated histogram CSV uses real activations captured from the CIFAR VTAB-style test split with the trained 3-bit QPEFT ViT-B/16 checkpoint:

`outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/20260114184203_default_acc=53.50/best_checkpoint.pth`

The default export uses block 5 and 32 CIFAR test images. It captures the input tensors to these quantizer modules:

| Figure panel | Hooked module |
|---|---|
| Ordinary activations | `blocks.5.attn.qkv.act_quantizer` |
| post-GELU activations | `blocks.5.mlp.fc2.act_quantizer` |
| post-softmax attention | `blocks.5.attn.attn_quantizer` |

The project imports `cupy` unconditionally inside the quantizer package. The export script provides a local CPU-backed fallback only for this analysis run when `cupy` is unavailable; it does not modify project source files.

## Reproduce

```bash
python analysis/figure3_activation_distributions/export_figure3_activation_histograms.py
python analysis/figure3_activation_distributions/plot_figure3_activation_distributions.py
```

The plotting script uses matplotlib only and exports SVG, PDF, and 600 dpi PNG. It can generate synthetic data only with `--demo`; demo output is not suitable for the final report.
