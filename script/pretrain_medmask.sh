# CUDA_VISIBLE_DEVICES=1 \
# python /home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrain_MedMask.py \
# --mask_ratio=0.7 \
# --intensity=1
# intensity 0 적용안함 / 1 적용함

#################
# 환경 변수 설정 #
#################
export nnUNet_raw="/nas_homes/yoonji/medmask/nnUNet_raw"
export nnUNet_preprocessed="/nas_homes/yoonji/medmask/nnUNet_preprocessed"
export nnUNet_results="/home/yoonji/AnatoMask/Anatomask_results"
#####################################################################

# CUDA_VISIBLE_DEVICES=1 \
# python /home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrain_MedMask_single.py \
# --mask_ratio=0.6 \
# --intensity=1 \
# --model_name="single_aleatory"

CUDA_VISIBLE_DEVICES=0 \
python /home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrain_MedMask.py \
--mask_ratio=0.6 \
--intensity=0 \
--model_name="medmask_1208_foreground"
