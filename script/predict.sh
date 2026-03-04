#!/bin/bash

export nnUNet_raw='/nas_homes/yoonji/medmask/nnUNet_raw'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/nas_homes/yoonji/medmask/nnUNet_results'

# DATA_PATH='/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs'
# CHECK_PATH="/nas_homes/yoonji/medmask/nnUNet_results/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres"
# LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs"

# # 배열 이름을 MODEL_NAMES로 변경하고 올바른 배열 문법 사용
# MODEL_NAMES=(
#     "medmask_0.6_1000epoch_0527"
#     # "medmask_0.6_1000epoch_0602_nointnesity"
#     # "medmask_0.7_1000epoch_0527"
#     # "medmask_0.8_1000epoch_0527"
#     # "spark_1000epoch"
#     # "anatomask_1000epoch"
# )

# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#     echo "Running inference for model: $MODEL_NAME"
    
#     CUDA_VISIBLE_DEVICES=1 \
#     python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new_calibration.py \
#                         --inp "$DATA_PATH" \
#                         --inp_label "$LABEL_PATH" \
#                         --checkpoint "$CHECK_PATH" \
#                         --model "$MODEL_NAME" \
#                         --num_classes 105
    
#     echo "Completed: $MODEL_NAME"
# done

DATA_PATH='/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs'
CHECK_PATH="/nas_homes/yoonji/medmask/nnUNet_results/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres"
LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs"

# 배열 이름을 MODEL_NAMES로 변경하고 올바른 배열 문법 사용
MODEL_NAMES=(
    "medmask_0.6_1000epoch_0527"
    # "medmask_0.6_1000epoch_0602_nointnesity"
    # "medmask_0.7_1000epoch_0527"
    # "medmask_0.8_1000epoch_0527"
    # "spark_1000epoch"
    # "anatomask_1000epoch"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running inference for model: $MODEL_NAME"
    
    CUDA_VISIBLE_DEVICES=1 \
    python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new_calibration.py \
                        --inp "$DATA_PATH" \
                        --inp_label "$LABEL_PATH" \
                        --checkpoint "$CHECK_PATH" \
                        --model "$MODEL_NAME" \
                        --num_classes 105
    
    echo "Completed: $MODEL_NAME"
done