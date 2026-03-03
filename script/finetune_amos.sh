####################################################################
# 경로 설정 #
############
# 1) nnUNetTrainer.py 의 result path 바꾸기
# 2) runtraining.py 의 checkpoint file 바꾸기 (이어서 돌릴 시)

#################
# 환경 변수 설정 #
#################
export nnUNet_raw="/nas_homes/yoonji/medmask/nnUNet_raw"
export nnUNet_preprocessed="/nas_homes/yoonji/medmask/nnUNet_preprocessed"
export nnUNet_results="/mnt/HDD/yoonji/medmim/nnUNet_results/"
#####################################################################
export CUDA_VISIBLE_DEVICES=3


#### TotalSegmentator ######
# (epoch 0부터)
# python nnunetv2/run/run_finetuning_STUNet.py Dataset606_all_TotalSegmentator 3d_fullres 0 \
# -pretrained_weights /nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.8/medmask_checkpoint_epoch_1000.pt \
# -tr STUNetTrainer \
# -device cuda \

# 이이서 돌리기
# export CUDA_VISIBLE_DEVICES=1
# python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
# -tr STUNetTrainer \
# -device cuda \
# --c \

###### Amos 22 ####  0617
# (epoch 0부터)
# python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
# -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.6/medmask_checkpoint_epoch_1000_nointensity.pt" \
# -tr STUNetTrainer \
# -device cuda \
# --dataset_name Dataset219_AMOS2022_postChallenge_task2 \
# --result_folder medmask_0.6_intensityfalse_0616

# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset309_FLARE22 \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.6/medmask_checkpoint_epoch_1000_nointensity.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset309_FLARE22 \
#     --result_folder medmask_0.6_intensityfalse_0616

python nnunetv2/run/run_finetuning_STUNet.py \
    Dataset219_AMOS2022_postChallenge_task2 \
    3d_fullres \
    0 \
    -tr STUNetTrainer \
    -device cuda \
    -num_gpus 1 \
    --c \
    --dataset_name Dataset219_AMOS2022_postChallenge_task2 \
    --result_folder medmask_0.6_intensityfalse_0616



# 이어서 돌리기
# export CUDA_VISIBLE_DEVICES=2
# python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
# -tr STUNetTrainer \
# -device cuda \
# --c \



############ 0616 ###########
# ## (0605) ###
# # device 0 [amos 돌리기 - medmask_0.6]
# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset219_AMOS2022_postChallenge_task2 \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.6/intensity_False/medmask_checkpoint_epoch_400.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset219_AMOS2022_postChallenge_task2 \
#     --result_folder medmask_0.6_intensityfalse_0616

# python nnunetv2/run/run_finetuning_STUNet.py \
#     Dataset309_FLARE22 \
#     3d_fullres \
#     0 \
#     -pretrained_weights "/nas_homes/yoonji/medmask/nnUNet_results/pretraining/medmask/0.6/intensity_False/medmask_checkpoint_epoch_400.pt" \
#     -tr STUNetTrainer \
#     -device cuda \
#     --dataset_name Dataset309_FLARE22 \
#     --result_folder medmask_0.6_intensityfalse_0616