# Figure 5 CIFAR Case-Study Training Trajectory

This directory contains the reproducible plotting code for Figure 5.

## Data source

The main figure is generated from the TensorBoard scalar tag `Val/epoch_acc1` in:

`outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/20260114184203_default_acc=53.50/tensorboard/events.out.tfevents.1768387332.master-192-168-8-48.1680.0`

The TensorBoard screenshot at `tensorboard.png` is recorded as a visual reference only. Numeric values are read from TensorBoard scalar data or the exported scalar CSV, not from the screenshot.

## Reproduce

```bash
python analysis/cifar_case_study_figure5/plot_figure5_cifar_acc1.py --prefer-event
```

The script also supports the exported scalar CSV:

```bash
python analysis/cifar_case_study_figure5/plot_figure5_cifar_acc1.py \
  --csv analysis/tensorboard_cifar_3bit_qpeft/csv/val_acc1.csv
```

The figure uses matplotlib only and exports SVG, PDF, and 600 dpi PNG. Caption wording:

`CIFAR case-study training trajectory for 3-bit QPEFT.`

This figure is a CIFAR case-study trajectory and should not be described as multi-task convergence evidence.
