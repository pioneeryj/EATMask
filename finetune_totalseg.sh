############
# 경로 설정 #
############
# nnunetv2.training.nnUNetTrainer.nnUNetTrainer 


# 1) nnUNetTrainer.py 에 어디에 저장할지지
# 2) runtraining.py 에 checkpoint 어디서 불러올지

#################
# 환경 변수 설정 #
#################
export nnUNet_raw="/nas_homes/yoonji/medmask/nnUNet_raw"
export nnUNet_preprocessed="/nas_homes/yoonji/medmask/nnUNet_preprocessed"
export nnUNet_results="/nas_homes/yoonji/medmask/nnUNet_results"
#####################################################################


export CUDA_VISIBLE_DEVICES=1

# 방법 1: pretrained_weights 없이 실행
python nnunetv2/run/run_finetuning_STUNet.py \
    Dataset606_all_TotalSegmentator \
    3d_fullres \
    0 \
    -tr STUNetTrainer \
    -device cuda \
    -num_gpus 1 \
    --c \
    --dataset_name Dataset606_all_TotalSegmentator \
    --result_folder medmask_0.6_1000epoch_0514

# 방법 2: pretrained_weights와 함께 실행하려면 아래 주석을 해제하고 위 코드는 주석 처리
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset606_all_TotalSegmentator \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.7/medmask_checkpoint_epoch_1000.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --c \
#     --dataset_name Dataset606_all_TotalSegmentator \
#     --result_folder medmask_0.6_1000epoch_0514
