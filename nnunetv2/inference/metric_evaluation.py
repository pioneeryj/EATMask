import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import monai
'''
Assume that,,
prediction_file: ~~.nii.gz
groundtruth_file: ~~.nii.gz
shape of image: (112,112,128) # H,W,D
'''
pred_dir = "/home/yoonji/AnatoMask/Anatomask_results/Dataset601_Total/Pretraining/anatomask/1000epoch"
label_dir = "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset601_organs_TotalSegmentator/labelsTs"

def get_file_list(dir_path):
    file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    if not file_list:
        raise FileNotFoundError(f"{dir_path} 내에 파일이 없습니다.")
    return file_list

def center_crop(image, target_size):
    """
    중앙 크롭 (image: (H, W, D), target_size: (112, 112, 128))
    """
    H, W, D = image.shape
    crop_H, crop_W, crop_D = target_size

    # 중앙 기준 크롭 시작 위치 계산
    start_H = max((H - crop_H) // 2, 0)
    start_W = max((W - crop_W) // 2, 0)
    start_D = max((D - crop_D) // 2, 0)

    # 크롭 종료 위치 계산
    end_H = min(start_H + crop_H, H)
    end_W = min(start_W + crop_W, W)
    end_D = min(start_D + crop_D, D)

    return image[start_H:end_H, start_W:end_W, start_D:end_D]

def pad_to_target_size(image, target_size):
    """
    부족한 크기를 패딩 (image: (h, w, d), target_size: (112, 112, 128))
    """
    h, w, d = image.shape
    pad_h = max(target_size[0] - h, 0)
    pad_w = max(target_size[1] - w, 0)
    pad_d = max(target_size[2] - d, 0)

    # 앞뒤 균등하게 패딩 분배
    pad_H = (pad_h // 2, pad_h - pad_h // 2)
    pad_W = (pad_w // 2, pad_w - pad_w // 2)
    pad_D = (pad_d // 2, pad_d - pad_d // 2)

    # F.pad의 pad 순서는 (pad_left_d, pad_right_d, pad_left_w, pad_right_w, pad_left_h, pad_right_h)
    return F.pad(image, (pad_D[0], pad_D[1], pad_W[0], pad_W[1], pad_H[0], pad_H[1]), mode='constant', value=0)


def segment_to_onehot(predicted_tensor, num_classes):
    predicted_long = predicted_tensor.long()    
    onehot = F.one_hot(predicted_long, num_classes=num_classes)
    onehot = onehot.permute(3, 0, 1, 2).float()
    return onehot
        
    
    
def calculate_dice_score(predicted, ground_truth, num_classes):
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean")
    # Convert to PyTorch tensors
    predicted_tensor = torch.from_numpy(predicted)
    ground_truth_tensor = torch.from_numpy(ground_truth)

    predict_onehot = predicted_tensor.unsqueeze(0).unsqueeze(0)
    ground_truth = ground_truth_tensor.unsqueeze(0).unsqueeze(0)
    # Calculate Dice score
    dice_score = dice_metric(y_pred=predict_onehot, y=ground_truth)
    return dice_score.item()

def calculate_nsd_score(predicted, ground_truth, num_classes):
    nsd_metric = monai.metrics.SurfaceDiceMetric(class_thresholds=[1.0], include_background=False, reduction="mean")
    
    predicted_tensor = torch.from_numpy(predicted)
    ground_truth_tensor = torch.from_numpy(ground_truth)
    
    predicted_tensor = predicted_tensor.unsqueeze(0).unsqueeze(0)
    ground_truth_tensor = ground_truth_tensor.unsqueeze(0).unsqueeze(0)
    
    nsd_score = nsd_metric(y_pred=predicted_tensor, y=ground_truth_tensor)
    
    return nsd_score.item()


def evaluate_predictions(pred_dir, label_dir):
    """
    예측 폴더(pred_dir)와 정답 폴더(label_dir)의 3D 이미지들을 매칭하여,
    각 이미지에 대해 Dice 점수를 계산하고 평균 Dice를 반환.
    
    - label 이미지에 대해서는 중앙 크롭 및 패딩을 적용하여 target_size (112,112,128)로 맞춤.
    - 예측 이미지는 이미 (112,112,128) 크기라고 가정.
    - 파일 매칭은 label 파일 이름에서 ".nii.gz"를 "_0000_pred.nii.gz"로 변경하여 진행.
    """
    target_size = (112, 112, 128)
    label_files = get_file_list(label_dir)
    
    dice_scores = []
    
    for label_file in label_files:
        # 예측 파일명: label 파일명에서 ".nii.gz"를 "_0000_pred.nii.gz"로 변경
        pred_file = label_file.replace(".nii.gz", "_0000_pred.nii.gz")
        pred_path = os.path.join(pred_dir, pred_file)
        label_path = os.path.join(label_dir, label_file)
        
        # 파일이 존재하는지 체크
        if not os.path.exists(pred_path):
            print(f"Warning: {pred_path} not found. Skipping.")
            continue
        
        # 불러오기 (numpy array, shape: (H, W, D))
        pred_img = nib.load(pred_path).get_fdata()
        gt_img = nib.load(label_path).get_fdata()
        
        # 정답(ground truth) 이미지 처리: 중앙 크롭 후 부족한 부분 패딩
        cropped_gt = center_crop(gt_img, target_size)
        # pad_to_target_size는 torch.Tensor 입력을 받으므로 변환 후 다시 numpy로 변환
        padded_gt = pad_to_target_size(torch.from_numpy(cropped_gt), target_size).numpy()
        
        # Dice 점수 계산 (예측 이미지는 이미 target_size라 가정)
        dice = calculate_dice_score(pred_img, padded_gt)
        dice_scores.append(dice)
        print(f"{label_file}: Dice = {dice:.4f}")
        
    if dice_scores:
        avg_dice = np.nanmean(dice_scores)
    else:
        avg_dice = 0.0
    print(f"Average Dice Score: {avg_dice:.4f}")
    return avg_dice


if __name__== '__main__' :
      evaluate_predictions(pred_dir, label_dir)