# Figure 2 Layerwise Sensitivity Heatmap

This directory contains the reproducible data export and plotting code for Figure 2.

## Data source

The CSV is extracted from the TensorBoard scalar event:

`outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/20260114184203_default_acc=53.50/tensorboard/events.out.tfevents.1768387332.master-192-168-8-48.1680.1`

The sensitivity metric is the final logged mean Adam second moment (`exp_avg_sq`) for each quantizer module. This is the same statistic written by `set_dropout_from_grads` in `engines/engine_finetune.py`.

## Module mapping

| Heatmap column | TensorBoard scalar tag |
|---|---|
| qkv | `blocks.{i}.attn.qkv.weight_quantizer` |
| proj | `blocks.{i}.attn.proj.weight_quantizer` |
| fc1 | `blocks.{i}.mlp.fc1.weight_quantizer` |
| fc2 | `blocks.{i}.mlp.fc2.weight_quantizer` |
| post-GELU | `blocks.{i}.mlp.fc2.act_quantizer` |
| post-softmax | `blocks.{i}.attn.attn_quantizer` |

## Reproduce

```bash
python analysis/figure2_layerwise_sensitivity/export_figure2_sensitivity_from_tensorboard.py
python analysis/figure2_layerwise_sensitivity/plot_figure2_sensitivity_heatmap.py
```

The plotting script reads a CSV with the required columns `layer_index`, `module_name`, and `sensitivity_value`, uses matplotlib only, and exports SVG, PDF, and 600 dpi PNG.
