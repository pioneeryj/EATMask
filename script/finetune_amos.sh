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
export nnUNet_results="/nas_homes/yoonji/medmask/nnUNet_results"
#####################################################################

#### TotalSegmentator ######
# (epoch 0부터터)
# python nnunetv2/run/run_finetuning_STUNet.py Dataset606_all_TotalSegmentator 3d_fullres 0 \
# -pretrained_weights /home/yoonji/AnatoMask/Anatomask_results/MedMask/Dataset601_organs/Pretraining/medmask/medmask_checkpoint_epoch_1000.pt \
# -tr STUNetTrainer \
# -device cuda \

# 이이서 돌리기
export CUDA_VISIBLE_DEVICES=1
python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
-tr STUNetTrainer \
-device cuda \
--c \

###### Amos 22 #### 
# (epoch 0부터)
# export CUDA_VISIBLE_DEVICES=2
# python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
# -pretrained_weights /home/yoonji/AnatoMask/Anatomask_results/MedMask/Dataset601_organs/Pretraining/medmask/medmask_checkpoint_epoch_1000.pt \
# -tr STUNetTrainer \
# -device cuda \

# 이어서 돌리기
# export CUDA_VISIBLE_DEVICES=2
# python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
# -tr STUNetTrainer \
# -device cuda \
# --c \
