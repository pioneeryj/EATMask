#!/bin/bash

export nnUNet_raw='/nas_homes/yoonji/medmask/nnUNet_raw'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/nas_homes/yoonji/medmask/nnUNet_results'
export CUDA_VISIBLE_DEVICES=3
DATA_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset219_AMOS2022_postChallenge_task2/imagesVa"
LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset219_AMOS2022_postChallenge_task2/labelsVa"

CHECK_PATH="/mnt/HDD/yoonji/medmim/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/STUNetTrainer__nnUNetPlans__3d_fullres"
MODEL_NAME="medmask_0.6_intensityfalse_0616"

PTH_NAMES=(
    "checkpoint_best.pth"
    "checkpoint_epoch_300.pth"
    "checkpoint_epoch_400.pth"
    "checkpoint_epoch_500.pth"
    "checkpoint_epoch_600.pth"
    "checkpoint_epoch_700.pth"
)

for PTH_NAME in "${PTH_NAMES[@]}"; do
    echo "Running inference for model: $MODEL_NAME"
    
    python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_single.py \
                        --inp "$DATA_PATH" \
                        --inp_label "$LABEL_PATH" \
                        --checkpoint "$CHECK_PATH" \
                        --model "$MODEL_NAME" \
                        --pth "$PTH_NAME" \
                        --num_classes 16
    
    echo "Completed: $MODEL_NAME"
done
