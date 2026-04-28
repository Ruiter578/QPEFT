#!/usr/bin/env bash
# set -x

# === ENV ===
source ~/anaconda3/bin/activate PYRA
conda info -e

PROJECT_ROOT=/datadisk/lisheng/projects/2024/AdaptFusion/IPESelfCalibrate_SCAdapter
EXEC_SCRIPT=./lib/save_pretrasformed_testset.py
GPU_ID=4

cd ${PROJECT_ROOT}

export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
# === ENV ===

# === EXP CFGS ===
# CONFIG="ViT-L_prompt_lora_12.yaml"
# CONFIG="ViT-B_prompt_lora_12.yaml"
CONFIG="ViT-B_prompt_lora_8"
CONFIG_DIR="experiments/LoRA/${CONFIG}.yaml"
# CKPT="weights/imagenet21k_ViT-L_16.npz"
CKPT="weights/ViT-B_16.npz"
WEIGHT_DECAY=0.0001

PYRA_LR=(0.0001)                       # 0.00001 0.00003 0.0001 0.0003 0.001 0.003
# PYRA_LR=(0.00001 0.00003 0.0001 0.0003 0.001 0.003)                       # 0.00001 0.00003 0.0001 0.0003 0.001 0.003
merge_schedule="high"
# merge_schedule="low"
DATASETS=(cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele)
# DATASETS=(svhn sun397 patch_camelyon eurosat resisc45)
# DATASETS=(diabetic_retinopathy clevr_count clevr_dist dmlab kitti)
# DATASETS=(dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele)

# DATASETS=(cifar)
# === EXP CFGS ===

currenttime=`date "+%Y%m%d_%H%M%S"`
for pyra_lr in "${PYRA_LR[@]}"
do
    for DATASET in "${DATASETS[@]}"
    do
        for LR in 0.001
        do
            LOG_DIR=outputs/${CONFIG}/with-sc-loss/${DATASET}/${merge_schedule}/${currenttime}_SC
            
            TARGET_DIR=${LOG_DIR}/lr-${LR}_wd-${WEIGHT_DECAY}_sep-lr=${pyra_lr}

            CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ${EXEC_SCRIPT} --data-path=./data/vtab-1k --data-set=${DATASET}\
                    --cfg=${CONFIG_DIR} --resume=${CKPT} --output_dir=${TARGET_DIR}\
                    --batch-size=64 --lr=${LR} --epochs=100 --weight-decay=${WEIGHT_DECAY}\
                    --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0\
                    --token_merging --merging_schedule=${merge_schedule}\
                    --pyra --separate_lr_for_pyra --pyra_lr=${pyra_lr} --enable_sc_loss
        done
    done
done