#!/bin/bash

export nnUNet_raw='/mnt/HDD/yoonji/medmim'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/mnt/HDD/yoonji/medmim/nnUNet_results'
export CUDA_VISIBLE_DEVICES=1

DATA_PATH="/mnt/HDD/yoonji/medmim/flare_dataset/imagesTs/sup_images"
LABEL_PATH="/mnt/HDD/yoonji/medmim/flare_dataset/labelsTs/labels"

CHECK_PATH="/mnt/HDD/yoonji/medmim/nnUNet_results/Dataset309_FLARE22/STUNetTrainer__nnUNetPlans__3d_fullres"
MODEL_NAME="anatomask_pre500"

PTH_NAMES=(
    "checkpoint_epoch_400.pth"
    # "checkpoint_epoch_600.pth"
    # "checkpoint_epoch_700.pth"
    # "checkpoint_epoch_800.pth"
    # "checkpoint_epoch_1000.pth"
)

for PTH_NAME in "${PTH_NAMES[@]}"; do
    echo "Running inference for model: $MODEL_NAME"
    
    CUDA_VISIBLE_DEVICES=3 \
    python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_single.py \
                        --inp "$DATA_PATH" \
                        --inp_label "$LABEL_PATH" \
                        --checkpoint "$CHECK_PATH" \
                        --model "$MODEL_NAME" \
                        --pth "$PTH_NAME" \
                        --num_classes 14
    
    echo "Completed: $MODEL_NAME"
done

