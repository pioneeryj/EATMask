#################
# 환경 변수 설정 #
#################
export nnUNet_raw="/nas_homes/yoonji/medmask/nnUNet_raw"
export nnUNet_preprocessed="/nas_homes/yoonji/medmask/nnUNet_preprocessed"
# export nnUNet_results="/nas_homes/yoonji/medmask/nnUNet_results"
export nnUNet_results="/mnt/HDD/yoonji/medmim/nnUNet_results/"
######################################
export CUDA_VISIBLE_DEVICES=1

# 이어서돌리기
python nnunetv2/run/run_finetuning_STUNet.py \
    Dataset309_FLARE22 \
    3d_fullres \
    0 \
    -tr STUNetTrainer \
    -device cuda \
    --c \
    --dataset_name Dataset309_FLARE22 \
    --result_folder anatomask_pre500


# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset219_AMOS2022_postChallenge_task2 \
#     3d_fullres \
#     0 \
#     -tr STUNetTrainer \
#     -device cuda \
#     --c \
#     --dataset_name Dataset219_AMOS2022_postChallenge_task2 \
#     --result_folder anatomask_pre500


# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset606_all_TotalSegmentator \
#     3d_fullres \
#     0 \
#     -tr STUNetTrainer \
#     -device cuda \
#     --c \
#     --dataset_name Dataset606_all_TotalSegmentator \
#     --result_folder anatomask_pre500