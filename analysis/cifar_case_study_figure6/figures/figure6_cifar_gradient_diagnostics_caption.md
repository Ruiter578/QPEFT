Figure 6. CIFAR case-study gradient diagnostics for 3-bit QPEFT. The panels show representative gradient-norm traces for Grad_attn_quantizer.log_base_alpha, Grad_qkv.lora_B, and Grad_mlp.fc2.lora_B in blocks 0, 5, and 11, plotted as log10 gradient norms. Thin traces are raw TensorBoard scalar values and thick traces are EMA-smoothed curves with span 81 for readability. This figure is a CIFAR case-study diagnostic rather than broad statistical validation.

- Data source: TensorBoard events under outputs_dev/vitb16/T=vtab_Q=3bit_L=lora_2_qkv-fc1-proj-fc2/cifar/20260114184203_default_acc=53.50/tensorboard
- Plotted scalar rows: 28800
- Rendering: matplotlib; no seaborn and no AI-generated curves.
