#!/bin/bash

export nnUNet_raw='/nas_homes/yoonji/medmask/nnUNet_raw'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/nas_homes/yoonji/medmask/nnUNet_results'

DATA_PATH="/mnt/HDD/yoonji/medmim/flare_dataset/imagesTs/sup_images"
LABEL_PATH="/mnt/HDD/yoonji/medmim/flare_dataset/labelsTs/labels"
CHECK_PATH="/mnt/HDD/yoonji/medmim/nnUNet_results/Dataset309_FLARE22/STUNetTrainer__nnUNetPlans__3d_fullres"

# DATA_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs"
# LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs"
# CHECK_PATH="/nas_homes/yoonji/medmask/nnUNet_results/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres"
# MODEL_NAME="medmask_0.6_1000epoch_0602_nointnesity"

# spark
MODEL_NAME="medmask_singleepistemic_0616"
PTH="checkpoint_epoch_400.pth"

# anatomask
# MODEL_NAME="anatomask_1000epoch_0605"
# PTH="checkpoint_epoch_500.pth"
# medmask

# amos
# medmask: "/nas_homes/yoonji/medmask/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/STUNetTrainer__nnUNetPlans__3d_fullres/medmask_0.6_intensityfalse_0616/checkpoint_best.pth"
# anatomask: "/nas_homes/yoonji/medmask/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/STUNetTrainer__nnUNetPlans__3d_fullres/anatomask_pre500/checkpoint_epoch_400.pth"
# spark: "/nas_homes/yoonji/medmask/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/STUNetTrainer__nnUNetPlans__3d_fullres/spark_1000epoch/checkpoint_epoch_300.pth"






CUDA_VISIBLE_DEVICES=2 \
python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_image.py \
                    --inp "$DATA_PATH" \
                    --inp_label "$LABEL_PATH" \
                    --checkpoint "$CHECK_PATH" \
                    --model "$MODEL_NAME" \
                    --pth "$PTH" \
                    --num_classes 14


# visualize 할 것들: 
# totalseg의 spark, anatomask, medmask, gt version
# amos의 spark, anatomask, medmask, gt version
# flare의 spark, anatomask, medmask, gt version