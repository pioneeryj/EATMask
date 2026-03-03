CUDA_VISIBLE_DEVICES=0 \
python /home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrain_MedMask_single.py \
--mask_ratio=0.6 \
--intensity=0 \
--model_name="nonintensity_epistemic"
