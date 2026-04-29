# Figure 6: CIFAR Gradient Diagnostics

This directory contains the final Figure 6 generation scripts for the CIFAR
3-bit QPEFT case-study diagnostic. The final report outputs are generated with
the matplotlib script.

The final figure is generated from real TensorBoard gradient scalar logs, not
from AI-generated or synthetic curves. The script also supports the exported
long-form CSV under `analysis/tensorboard_cifar_3bit_qpeft/csv/`.

## Reproduce

```bash
python analysis/cifar_case_study_figure6/plot_figure6_cifar_gradient_diagnostics_matplotlib.py --prefer-events
```

Outputs are written to `analysis/cifar_case_study_figure6/figures/`:

- `figure6_cifar_gradient_diagnostics.svg`
- `figure6_cifar_gradient_diagnostics.pdf`
- `figure6_cifar_gradient_diagnostics.png` with 600 dpi metadata
- `figure6_cifar_gradient_diagnostics_caption.md`
- `figure6_cifar_gradient_diagnostics_summary.csv`
- `figure6_cifar_gradient_diagnostics_plotted_points.csv`
- `figure6_cifar_gradient_diagnostics_gpt_image2_reference.png`

The `*_gpt_image2_reference.png` file is a visual-polishing reference generated
after the reproducible matplotlib figure. It should not replace the SVG/PDF/PNG
matplotlib outputs as the quantitative source-of-truth figure, because image
generation can alter exact curve geometry.

## Caption

Use the caption in `figures/figure6_cifar_gradient_diagnostics_caption.md`.
It explicitly states that this is a CIFAR case-study diagnostic rather than
broad statistical validation.
