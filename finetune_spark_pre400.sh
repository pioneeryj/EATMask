#################
# 환경 변수 설정 #
#################
export nnUNet_raw="/nas_homes/yoonji/medmask/nnUNet_raw"
export nnUNet_preprocessed="/nas_homes/yoonji/medmask/nnUNet_preprocessed"
# export nnUNet_results="/nas_homes/yoonji/medmask/nnUNet_results"
export nnUNet_results="/mnt/HDD/yoonji/medmim/nnUNet_results/"

export CUDA_VISIBLE_DEVICES=2

# totalseg
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset606_all_TotalSegmentator \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/spark/Spark_checkpoint_epoch_400.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset606_all_TotalSegmentator \
#     --result_folder spark_pre400
    
# amos
python nnunetv2/run/run_finetuning_STUNet.py \
    Dataset219_AMOS2022_postChallenge_task2 \
    3d_fullres \
    0 \
    -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/spark/Spark_checkpoint_epoch_400.pt" \
    -tr STUNetTrainer \
    -device cuda \
    --dataset_name Dataset219_AMOS2022_postChallenge_task2 \
    --result_folder spark_pre400

# # flare
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset309_FLARE22 \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/spark/Spark_checkpoint_epoch_400.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset309_FLARE22 \
#     --result_folder spark_pre400