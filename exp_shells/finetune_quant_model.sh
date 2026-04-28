#!/usr/bin/env bash
# set -x
set -euo pipefail
# === ENV ===
source /opt/conda/etc/profile.d/conda.sh
conda activate PEFT

conda info -e

PROJECT_ROOT=/TRS-SAS/linwei/direct_quant_dev_clean
GPU_ID=2

EXEC_SCRIPT=main_vtab.py
USE_CORES=10

DATASET=cifar
MODEL=vitb16
TRAIN_CFGS="vtab"
QUANT_CFGS="3bit"
# QUANT_CFGS="4bit_w-tensor-wise"
LORA_CFGS="lora_2_qkv-fc1-proj-fc2"

LOG_ROOT="outputs_dev"

EXP_NAME="default"


cd ${PROJECT_ROOT}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${GPU_ID}
currenttime=`date "+%Y%m%d_%H%M%S"`

total_cores=$(nproc)
selected_cores=$(shuf -i 0-$(($total_cores - 1)) -n ${USE_CORES} | tr '\n' ',' | sed 's/,$//')

taskset -c ${selected_cores} python -u ${EXEC_SCRIPT} --model ${MODEL}\
 --train_cfgs ${TRAIN_CFGS} --lora_cfgs ${LORA_CFGS} --quant_cfgs ${QUANT_CFGS}\
 --dataset ${DATASET} --log_root ${LOG_ROOT} --save_ckpt --exp_name ${EXP_NAME}
