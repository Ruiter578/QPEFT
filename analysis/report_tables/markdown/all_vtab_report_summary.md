| Workbook | Sheet | Method | W/A | Params | Natural Avg | Specialized Avg | Structured Avg | Group Avg | Tot. Avg | Comments | Anomaly Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| qpeft_exps.xlsx | vitb_vtab | LoRA(r=2),qkv_fc1 | 32/32 | 165888 | 81.34 | 86.14 | 60.30 | 75.93 | 73.49 |  |  |
| qpeft_exps.xlsx | vitb_vtab | LoRA(r=2),qkv_fc1,with-grad-scale | 4/4 | 165888+83484 | 75.78 | 85.36 | 60.14 | 73.76 | 71.21 | init shift=0.17, per-tensor |  |
| qpeft_exps.xlsx | vitb_vtab | LoRA(r=2),qkv_fc1,with-grad-scale | 3/3 | 165888+83484 | 74.18 | 84.59 | 58.92 | 72.56 | 69.95 | init shift=0.17, per-tensor |  |
| qpeft_exps.xlsx | vitl_vtab | lora2, qkv-fc1 | 32/32 | 442368 | 83.35 | 86.53 | 58.87 | 76.25 | 73.71 |  |  |
| qpeft_exps.xlsx | vitl_vtab | LoRA(r=2),qkv_fc1,with-grad-scale | 3/3 | 442368+222168 | 80.16 | 85.27 | 60.58 | 75.34 | 72.99 |  |  |
| qpeft_exps.xlsx | vitl_vtab | LoRA(r=2),qkv_fc1,with-grad-scale | 4/4 | 442368+222168 | 81.81 | 86.08 | 60.36 | 76.08 | 73.68 |  |  |
| qpeft_exps.xlsx | vitl_vtab | LoRA(r=2),qkv_fc1,with-grad-scale | 2/2 |  | 62.31 | 79.37 | 52.87 | 64.85 | 61.92 |  |  |
| qpeft_exps.xlsx | QAT_baselines | T=vtab_Q=3bit_L=None |  |  | 48.35 | 75.59 | 45.27 | 56.40 | 52.79 |  |  |
| qpeft_exps.xlsx | QAT_baselines | T=vtab_Q=4bit_L=None |  |  | 60.52 | 75.91 | 46.43 | 60.95 | 57.83 |  |  |
| qpeft_exps.xlsx | abl_on_shift | no-shift | 3/3 | 165888+83328 | 65.92 | 81.75 | 54.77 | 67.48 | 64.56 | 没有任何shift |  |
| qpeft_exps.xlsx | abl_on_shift | shift as constant | 3/3 | 165888+83328 | 73.98 | 84.49 | 58.91 | 72.46 | 69.84 | post-gelu-shift=0.17 |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=0 | 3/3 | 165888+83340 | 61.99 | 82.43 | 57.12 | 67.18 | 64.24 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=01 |  |  | 71.47 | 83.71 | 57.90 | 71.03 | 68.33 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=02 |  |  | 73.32 | 84.10 | 59.27 | 72.23 | 69.67 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=03 |  |  | 73.90 | 84.36 | 58.90 | 72.39 | 69.79 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=04 |  |  | 74.11 | 84.34 | 59.28 | 72.58 | 70.02 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=05 |  |  | 74.28 | 84.14 | 59.38 | 72.60 | 70.08 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=06 |  |  | 74.30 | 84.35 | 59.09 | 72.58 | 70.01 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=07 |  |  | 74.27 | 84.84 | 58.88 | 72.66 | 70.01 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=08 |  |  | 73.94 | 84.28 | 58.78 | 72.34 | 69.74 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=09 |  |  | 73.91 | 84.18 | 57.85 | 71.98 | 69.31 |  |  |
| qpeft_exps.xlsx | abl_on_drop_epc | drop_epc=10 |  |  | 73.70 | 84.11 | 57.86 | 71.89 | 69.22 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | FP32 fine-tuned ViT | 32bit |  | 81.34 | 86.14 | 60.30 | 75.93 | 73.49 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | Adalog | 3bit |  | 72.03 | 79.00 | 40.81 | 63.94 | 60.35 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | Adalog | 4-bit |  | 79.68 | 84.85 | 54.48 | 73.00 | 70.16 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | Adalog | 2bit |  | 8.17 | 49.33 | 14.29 | 23.93 | 19.41 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | Repq-vit | 4 bit |  | 76.18 | 82.92 | 41.41 | 66.84 | 62.96 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | Repq-vit | 3 bit |  | 16.99 | 47.09 | 16.63 | 26.90 | 23.18 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | PTQ4VIT | 4 bit |  | 75.63 | 80.90 | 43.70 | 66.75 | 63.30 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | PTQ4VIT | 3 bit |  | 60.54 | 70.86 | 32.58 | 54.66 | 50.94 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | PTQ4VIT | 2 bit |  | 3.97 | 30.59 | 11.69 | 15.41 | 12.82 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | Qdrop | 3bit |  | 66.31 | 81.23 | 47.96 | 65.17 | 61.72 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitb_vtab_234bit | Qdrop | 4bit |  | 74.07 | 83.45 | 53.46 | 70.33 | 67.37 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | FP32 fine-tuned ViT | 32bit |  | 81.34 | 86.14 | 60.30 | 75.93 | 73.49 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | Adalog | 4 bit |  | 68.43 | 84.45 | 52.54 | 68.47 | 65.11 |  | Suspicious percentage scale: Pets=0.9054; Natural median=71.60. Possibly 90.54. |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | Adalog | 3 bit |  | 63.26 | 54.60 | 31.45 | 49.77 | 48.05 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | Adalog | 2 bit |  | 4.73 | 21.34 | 12.43 | 12.83 | 11.47 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | REPQ-vit | 4 bit |  | 78.47 | 81.62 | 46.16 | 68.75 | 65.53 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | REPQ-vit | 3 bit |  | 35.70 | 55.32 | 17.87 | 36.30 | 32.32 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | PTQ4VIT | 4 bit |  | 16.47 | 43.59 | 22.97 | 27.68 | 24.92 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | PTQ4VIT | 3 bit |  | 9.07 | 35.67 | 14.87 | 19.87 | 17.11 |  |  |
| QPEFT_PTQ对照实验结果.xlsx | vitl_vtab_234bit | PTQ4VIT | 2 bit |  | 3.28 | 21.34 | 13.42 | 12.68 | 11.35 |  |  |