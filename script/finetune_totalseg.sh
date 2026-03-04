####################################################################
# 경로 설정 #
############
# nnunetv2.training.nnUNetTrainer.nnUNetTrainer 


# 1) nnUNetTrainer.py 의 result path 바꾸기 (새로운 variant일시)
# 2) runtraining.py 의 checkpoint file 바꾸기 (이어서 돌릴 시)

#################
# 환경 변수 설정 #
#################
export nnUNet_raw="/nas_homes/yoonji/medmask/nnUNet_raw"
export nnUNet_preprocessed="/nas_homes/yoonji/medmask/nnUNet_preprocessed"
export nnUNet_results="/nas_homes/yoonji/medmask/nnUNet_results"
#####################################################################

# total segmentator
# spark


# amos
export CUDA_VISIBLE_DEVICES=1 
python nnunetv2/run/run_finetuning_STUNet.py Dataset219_AMOS2022_postChallenge_task2 3d_fullres 0 \
-pretrained_weights /home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrained_model/large_ep4k.model \
-tr STUNetTrainer \
-device cuda \
