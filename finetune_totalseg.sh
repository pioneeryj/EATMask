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

# total segmentator
# spark
export CUDA_VISIBLE_DEVICES=0
python nnunetv2/run/run_finetuning_STUNet.py Dataset606_all_TotalSegmentator 3d_fullres 0 \
-tr STUNetTrainer \
-device cuda \
--c \


# amos
# export CUDA_VISIBLE_DEVICES=1 
# python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
# -pretrained_weights Anatomask_results/Dataset601_Total/Pretraining/0224/anatomask/anatomask_checkpoint_epoch_1000.pt \
# -tr STUNetTrainer \
# -device cuda \

