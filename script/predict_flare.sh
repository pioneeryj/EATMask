
#!/bin/bash

export nnUNet_raw='/nas_homes/yoonji/medmask/nnUNet_raw'
export nnUNet_preprocessed='/nas_homes/yoonji/medmask/nnUNet_preprocessed'
export nnUNet_results='/nas_homes/yoonji/medmask/nnUNet_results'

DATA_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset219_AMOS2022_postChallenge_task2/imagesVa"
LABEL_PATH="/nas_homes/yoonji/medmask/nnUNet_raw/Dataset219_AMOS2022_postChallenge_task2/labelsVa"

CHECK_PATH="/nas_homes/yoonji/medmask/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/STUNetTrainer__nnUNetPlans__3d_fullres"


MODEL_NAMES=(
    "medmask_0.6_1000epoch"
    "medmask_0.7_1000epoch"
    "medmask_0.8_1000epoch"
    "spark_1000epoch"
    "anatomask_1000epoch"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running inference for model: $MODEL_NAME"
    
    CUDA_VISIBLE_DEVICES=0 \
    python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new.py \
                        --inp "$DATA_PATH" \
                        --inp_label "$LABEL_PATH" \
                        --checkpoint "$CHECK_PATH" \
                        --model "$MODEL_NAME" \
                        --num_classes 16
    
    echo "Completed: $MODEL_NAME"
done
