#!/bin/bash

export nnUNet_raw='/nas_homes/yoonji/medmask/nnUNet_raw'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/nas_homes/yoonji/medmask/nnUNet_results'

DATA_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs"
LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs"
CHECK_PATH="/mnt/HDD/yoonji/medmim/nnUNet_results/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres"
MODEL_NAME="medmask_0.6_1000epoch_aleatory_only"
# MODEL_NAME="medmask_0.6_1000epoch_0602_nointnesity"
# MODEL_NAME="medmask_0.7_1000epoch_0527"
PTH_NAMES=(
    # "checkpoint_best.pth"
    "checkpoint_epoch_100.pth"
    "checkpoint_epoch_200.pth"
    # "checkpoint_epoch_600.pth"
    # "checkpoint_epoch_700.pth"
    # "checkpoint_epoch_800.pth"
)

for PTH_NAME in "${PTH_NAMES[@]}"; do
    echo "Running inference for model: $MODEL_NAME"
    
    CUDA_VISIBLE_DEVICES=1 \
    python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_single.py \
                        --inp "$DATA_PATH" \
                        --inp_label "$LABEL_PATH" \
                        --checkpoint "$CHECK_PATH" \
                        --model "$MODEL_NAME" \
                        --pth "$PTH_NAME" \
                        --num_classes 105
    
    echo "Completed: $MODEL_NAME"
done