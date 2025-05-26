#!/bin/bash

export nnUNet_raw='/nas_homes/yoonji/medmask/nnUNet_raw'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/nas_homes/yoonji/medmask/nnUNet_results'


DATA_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs"
LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs"

CHECK_PATH="/nas_homes/yoonji/medmask/nnUNet_results/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres"
MODEL_NAME="medmask_0.7_1000epoch"


CUDA_VISIBLE_DEVICES=1 \
python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_single.py \
                      --inp "$DATA_PATH" \
                      --inp_label "$LABEL_PATH" \
                      --checkpoint "$CHECK_PATH" \
                      --model "$MODEL_NAME" \
                      --num_classes 105 # Totalseg:105, Amos:16
