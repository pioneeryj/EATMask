#!/bin/bash

export nnUNet_raw='/nas_homes/yoonji/medmask/nnUNet_raw'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/nas_homes/yoonji/medmask/nnUNet_results'

# DATA_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset219_AMOS2022_postChallenge_task2/imagesVa"
# LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset219_AMOS2022_postChallenge_task2/labelsVa"
# CHECK_PATH="/nas_homes/oonji/medmask/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/STUNetTrainer__nnUNetPlans__3d_fullres"


DATA_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs"
LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs"
CHECK_PATH="/nas_homes/yoonji/medmask/nnUNet_results/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres"
MODEL_NAME="medmask_0.6_1000epoch_0527"

CUDA_VISIBLE_DEVICES=2 \
python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_image.py \
                    --inp "$DATA_PATH" \
                    --inp_label "$LABEL_PATH" \
                    --checkpoint "$CHECK_PATH" \
                    --model "$MODEL_NAME" \
                    --num_classes 105


