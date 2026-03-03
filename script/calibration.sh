
# anatomask
# python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new.py \
#     --checkpoint "/home/yoonji/AnatoMask/Anatomask_results/MedMask/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres/anatomask/checkpoint_best.pth" \
#     --output "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs_pred/anatomask" \
#     --label "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs" \
#     --dataset "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs"

#medmask/home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new_calibration.py
python /home/yoonji/AnatoMask/nnunetv2/inference/predict_from_raw_data_new_calibration.py \
    --checkpoint "/home/yoonji/AnatoMask/Anatomask_results/MedMask/Dataset606_all_TotalSegmentator/STUNetTrainer__nnUNetPlans__3d_fullres/medmask_1000epoch_0.6/checkpoint_best.pth" \
    --model "medmask_caliration" \
    --inp_label "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs" \
    --inp "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/imagesTs" \
    --num_classes 105