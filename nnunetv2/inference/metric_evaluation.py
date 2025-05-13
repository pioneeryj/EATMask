import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import monai
from monai.metrics import DiceMetric
from monai.metrics import SurfaceDiceMetric
from scipy.ndimage import binary_erosion, distance_transform_edt
from typing import List, Tuple, Union
'''
Assume that,,
prediction_file: ~~.nii.gz
groundtruth_file: ~~.nii.gz
shape of image: (112,112,128) # H,W,D
'''
# pred_dir = "/home/yoonji/AnatoMask/Anatomask_results/Dataset601_Total/Pretraining/anatomask/1000epoch"
# label_dir = "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset601_organs_TotalSegmentator/labelsTs"

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


 
def calculate_dice_score(predicted, ground_truth):

    # DiceMetric은 torch.Tensor를 입력으로 받으므로, numpy 배열을 tensor로 변환
    predicted_tensor = torch.from_numpy(predicted).float()
    ground_truth_tensor = torch.from_numpy(ground_truth).float()
    
    predicted_tensor = predicted_tensor.unsqueeze(0)
    ground_truth_tensor = ground_truth_tensor.unsqueeze(0).unsqueeze(0)
    
    num_classes = predicted.shape[0]
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True, num_classes=num_classes)
    
    dice_metric(y_pred=predicted_tensor, y=ground_truth_tensor)
    
    return dice_metric.aggregate()

# def calculate_nsd_score(predicted, ground_truth, threshold=1.0):
#     nsd = SurfaceDiceMetric(include_background=False, reduction="mean", get_not_nans=False, threshold=threshold)
#     nsd(y_pred=predicted, y=ground_truth)
#     nsd.aggregate()
#     mean_nsd = nsd.mean()
#     return mean_nsd.item()
def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    주어진 segmentation에서 특정 레이블 또는 레이블 그룹에 해당하는 마스크를 생성합니다.
    
    Args:
        segmentation: 분할 마스크 (h,w,d 차원)
        region_or_label: 하나의 레이블 인덱스 또는 레이블 인덱스의 튜플
        
    Returns:
        생성된 불리언 마스크
    """
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    """
    True Positive, False Positive, False Negative, True Negative를 계산합니다.
    
    Args:
        mask_ref: 참조 마스크
        mask_pred: 예측 마스크
        ignore_mask: 무시할 픽셀의 마스크 (선택적)
        
    Returns:
        tp, fp, fn, tn 값
    """
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

def compute_dice_score(label: np.ndarray, prediction: np.ndarray, 
                      classes: List[int], ignore_label: int = None) -> dict:
    """
    3D 의료 영상의 각 클래스에 대한 Dice score를 계산합니다.
    
    Args:
        label: 정답 레이블 (h,w,d 차원)
        prediction: 예측 결과 (h,w,d 차원)
        classes: 평가할 클래스 인덱스 목록
        ignore_label: 평가에서 제외할 레이블 (선택적)
        
    Returns:
        각 클래스의 Dice score와 평균 점수를 포함하는 딕셔너리
    """
    ignore_mask = label == ignore_label if ignore_label is not None else None
    
    results = {}
    for cls in classes:
        mask_ref = region_or_label_to_mask(label, cls)
        mask_pred = region_or_label_to_mask(prediction, cls)
        
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        
        if tp + fp + fn == 0:
            dice = float('nan')  # 클래스가 존재하지 않는 경우
        else:
            dice = 2 * tp / (2 * tp + fp + fn)
            
        results[f'class_{cls}'] = {
            'Dice': float(dice),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'n_pred': int(fp + tp),  # 예측에서 해당 클래스의 총 복셀 수
            'n_ref': int(fn + tp),   # 참조에서 해당 클래스의 총 복셀 수
        }
    
    # 원본 코드처럼 지표 목록을 가져와서 처리
    metric_list = list(results[f'class_{classes[0]}'].keys())
    
    # 각 클래스별 평균 지표 (NaN 값 제외)
    class_means = {}
    for cls in classes:
        class_means[cls] = {}
        for metric in metric_list:
            class_means[cls][metric] = results[f'class_{cls}'][metric]
    
    # 전경(배경 제외) 클래스에 대한 평균 지표 (NaN 값 제외)
    foreground_mean = {}
    for metric in metric_list:
        values = []
        for cls in classes:
            # 배경(클래스 0) 제외하고 전경 클래스만 계산
            if cls == 0 or str(cls) == '0':
                continue
            if not np.isnan(class_means[cls][metric]):  # NaN 값 제외
                values.append(class_means[cls][metric])
        # NaN을 제외한 평균 계산
        foreground_mean[metric] = float(np.mean(values)) if values else float('nan')
    
    return {
        'metrics_per_class': results,
        'class_means': class_means,
        'foreground_mean': foreground_mean
    }
    
def calculate_dice_score_manual(label: np.ndarray, prediction: np.ndarray, 
                            classes: List[int] = None, ignore_label: int = None) -> dict:
    """
    3D 의료 영상 분할 모델의 성능을 평가합니다.
    
    Args:
        label: 정답 레이블 (h,w,d 차원)
        prediction: 예측 결과 (h,w,d 차원)
        classes: 평가할 클래스 인덱스 목록 (None이면 자동 감지)
        ignore_label: 평가에서 제외할 레이블 (선택적)
        
    Returns:
        평가 지표를 포함하는 딕셔너리
    """
    # 입력 검사
    assert label.shape == prediction.shape, "Label과 prediction의 차원이 일치해야 합니다."
    assert len(label.shape) == 3, "입력은 3차원(h,w,d) 배열이어야 합니다."
    
    # 클래스를 자동으로 감지
    if classes is None:
        classes = np.unique(label)
        if ignore_label is not None and ignore_label in classes:
            classes = np.delete(classes, np.where(classes == ignore_label))
    
    # Dice score 계산
    dice_results = compute_dice_score(label, prediction, classes, ignore_label)
    
    return dice_results

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