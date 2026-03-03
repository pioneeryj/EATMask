export nnUNet_raw="/nas_homes/yoonji/medmask/nnUNet_raw"
export nnUNet_preprocessed="/nas_homes/yoonji/medmask/nnUNet_preprocessed"
export nnUNet_results="/mnt/HDD/yoonji/medmim/nnUNet_results/"
############################################################################
DATASETS=(
    #"Dataset606_all_TotalSegmentator"
    "Dataset309_FLARE22"
    #"Dataset219_AMOS2022_postChallenge_task2"
)
################################################################################
# for DATASET in "${DATASETS[@]}"; do
#     echo "Running finetuning for model: $DATASET"
    
#     CUDA_VISIBLE_DEVICES=0 \
#     python nnunetv2/run/run_finetuning_STUNet.py \
#         "$DATASET" \
#         3d_fullres \
#         0 \
#         -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/single_aleatory/0.6/single_aleatory_checkpoint_epoch_500.pt" \
#         -tr STUNetTrainer \
#         -device cuda \
#         --dataset_name "$DATASET" \
#         --result_folder medmask_singlealeatory_0616
    
#     echo "Completed: $DATASET"
# done

for DATASET in "${DATASETS[@]}"; do
    echo "Running finetuning for model: $DATASET"
    
    CUDA_VISIBLE_DEVICES=1 \
    python nnunetv2/run/run_finetuning_STUNet.py \
        "$DATASET" \
        3d_fullres \
        0 \
        -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.6/intensity_False/medmask_checkpoint_epoch_400.pt" \
        -tr STUNetTrainer \
        -device cuda \
        --dataset_name "$DATASET" \
        --result_folder medmask_nonintensity_400_0622
    
    echo "Completed: $DATASET"
done

