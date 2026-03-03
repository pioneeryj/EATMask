
# anatomask
# python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new.py \
#     --checkpoint "/home/yoonji/AnatoMask/Anatomask_results/MedMask/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres/anatomask/checkpoint_best.pth" \
#     --output "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs_pred/anatomask" \
#     --label "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs" \
#     --dataset "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs"

#medmask
python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new.py \
    --checkpoint "/home/yoonji/AnatoMask/Anatomask_results/MedMask/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres/medmask_1000epoch_0.6/checkpoint_best.pth" \
    --output "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs_pred/medmask" \
    --label "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs" \
    --dataset "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs"