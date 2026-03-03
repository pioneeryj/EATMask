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
# export nnUNet_results="/nas_homes/yoonji/medmask/nnUNet_results"
export nnUNet_results="/mnt/HDD/yoonji/medmim/nnUNet_results/"

export CUDA_VISIBLE_DEVICES=0
#####################################################################

# 방법 1: pretrained_weights 없이 실행
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset606_all_TotalSegmentator \
#     3d_fullres \
#     0 \
#     -tr STUNetTrainer \
#     -device cuda \
#     -num_gpus 1 \
#     --c \
#     --dataset_name Dataset606_all_TotalSegmentator \
#     --result_folder medmask_0.7_1000epoch_0526

## 0602 0. Anatomask 처음부터 돌리기 (0.6)
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset606_all_TotalSegmentator \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/single_aleatory/0.6/single_aleatory_checkpoint_epoch_1000.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset606_all_TotalSegmentator \
#     --result_folder intensity_true_al_1000

# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset606_all_TotalSegmentator \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/nonintensity_epistemic/0.6/nonintensity_epistemic_head_latest.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset606_all_TotalSegmentator \
#     --result_folder single_ep_0620

python nnunetv2/run/run_finetuning_STUNet.py \
    Dataset606_all_TotalSegmentator \
    3d_fullres \
    0 \
    -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/nonintensity_aleatory/0.6/nonintensity_aleatory_head_latest.pt" \
    -tr STUNetTrainer \
    -device cuda \
    --dataset_name Dataset606_all_TotalSegmentator \
    --result_folder single_al_0620

# # 0602 1. Amos 처음부터 돌리기 (0.6)
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset219_AMOS2022_postChallenge_task2 \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.6/medmask_checkpoint_epoch_1000.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset219_AMOS2022_postChallenge_task2 \
#     --result_folder anatomask_1000epoch_0602

# # 0602 2. Totalseg nonintensity 돌리기 
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset606_all_TotalSegmentator \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/single_epistemic/0.6/single_epistemic_checkpoint_epoch_1000.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset606_all_TotalSegmentator \
#     --result_folder medmask_0.6_1000epoch_epistemic_only


## (0605) ###
# device 0 [flare22 돌리기 - medmask_0.6]
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset309_FLARE22 \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.6/intensity_False/medmask_checkpoint_epoch_400.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset309_FLARE22 \
#     --result_folder medmask_0.6_intensityfalse_0616

# device 0 [flare22 돌리기 - anatomask]
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset309_FLARE22 \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/anatomask/anatomask_checkpoint_epoch_1000.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset309_FLARE22 \
#     --result_folder anatomask_1000epoch_0605

# device 0 [flare22 이어서 돌리기 - medmask]
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset309_FLARE22 \
#     3d_fullres \
#     0 \
#     -tr STUNetTrainer \
#     -device cuda \
#     --c \
#     --dataset_name Dataset309_FLARE22 \
#     --result_folder medmask_0.6_1000epoch_0605
