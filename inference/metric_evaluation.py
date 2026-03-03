import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import monai
from monai.metrics import DiceMetric
from monai.metrics import SurfaceDiceMetric
from scipy.ndimage import binary_erosion, distance_transform_edt
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

# def calculate_dice_score(predicted: np.ndarray, ground_truth: np.ndarray, num_classes: None, ignore_empty: bool = True, eps: float = 1e-8) -> float:
#     """
#     3D segmentation map (H, W, D) numpy 배열에서 Dice 점수를 계산합니다.
    
#     Args:
#         predicted (np.ndarray): 모델 예측 결과로 각 voxel에 정수 레이블이 할당된 배열.
#         ground_truth (np.ndarray): GT로 각 voxel에 정수 레이블이 할당된 배열.
#         num_classes (int): 전체 클래스 수(백그라운드 포함). 기본값은 105.
#         ignore_empty (bool): True인 경우, ground truth 채널이 전부 0인 경우는 Dice 계산에서 배제합니다.
#         eps (float): 0으로 나누는 문제를 방지하기 위한 작은 값.
        
#     Returns:
#         float: non-empty 채널에 대해 계산된 평균 Dice score.
#     """
#     # 예측과 GT를 torch tensor로 변환 (정수형)
#     if num_classes is None:
#         num_pred_classes = predicted.max() + 1
#         num_gt_classes = ground_truth.max() + 1
#         num_classes_ = int(max(num_pred_classes, num_gt_classes))
#     else:
#         num_classes_ = int(num_classes)
#     pred_tensor = torch.from_numpy(predicted).long()
#     gt_tensor = torch.from_numpy(ground_truth).long()
    
#     # one-hot 인코딩: (H, W, D, num_classes)
#     pred_onehot = F.one_hot(pred_tensor, num_classes=num_classes_)
#     gt_onehot = F.one_hot(gt_tensor, num_classes=num_classes_)
    
#     # 채널을 가장 앞으로: (num_classes, H, W, D)
#     pred_onehot = pred_onehot.permute(3, 0, 1, 2).float()
#     gt_onehot = gt_onehot.permute(3, 0, 1, 2).float()
    
#     dice_per_class = []
    
#     # 각 채널별로 Dice 계산
#     for c in range(num_classes_):
#         p_c = pred_onehot[c]
#         g_c = gt_onehot[c]
        
#         # 교집합: 두 채널 모두 1인 voxel의 합
#         intersection = torch.sum(p_c * g_c)
#         # 예측과 GT에서 1인 voxel의 총합
#         sum_pred = torch.sum(p_c)
#         sum_gt = torch.sum(g_c)
        
#         if sum_gt.item() == 0:
#             # GT가 완전 비어있는 경우
#             if ignore_empty:
#                 continue  # Dice 계산에서 제외
#             else:
#                 # 예측도 비어있으면 Dice = 1, 아니면 0
#                 dice_c = 1.0 if sum_pred.item() == 0 else 0.0
#         else:
#             dice_c = (2.0 * intersection + eps) / (sum_pred + sum_gt + eps)
        
#         dice_per_class.append(dice_c)
    
#     # 만약 전부 skip된 경우
#     if not dice_per_class:
#         return float('nan')
    
#     mean_dice = sum(dice_per_class) / len(dice_per_class)
#     return mean_dice.item()

# def calculate_nsd_score(predicted, ground_truth, threshold=1.0, num_classes=None):
#     if num_classes is None:
#         num_pred_classes = predicted.max() + 1
#         num_gt_classes = ground_truth.max() + 1
#         num_classes_ = int(max(num_pred_classes, num_gt_classes))
#     else:
#         num_classes_ = int(num_classes)
    
#     nsd = np.empty(num_classes_, dtype=np.float32)
#     nsd.fill(np.nan)
    
#     # 3D 이미지의 연결구조: 모든 이웃 (26-connected)
#     structure = np.ones((3, 3, 3), dtype=bool)
    
#     # 각 클래스별로 NSD 계산
#     for c in range(num_classes_):
#         # 클래스 c에 대한 예측 및 GT binary mask 생성
#         pred_mask = (predicted == c)
#         gt_mask = (ground_truth == c)
        
#         # predicted boundary: 원본 mask에서 binary erosion을 수행한 결과와의 차집합
#         if np.any(pred_mask):
#             pred_eroded = binary_erosion(pred_mask, structure=structure, border_value=0)
#             pred_boundary = pred_mask & (~pred_eroded)
#         else:
#             pred_boundary = np.zeros_like(pred_mask, dtype=bool)
        
#         # GT boundary
#         if np.any(gt_mask):
#             gt_eroded = binary_erosion(gt_mask, structure=structure, border_value=0)
#             gt_boundary = gt_mask & (~gt_eroded)
#         else:
#             gt_boundary = np.zeros_like(gt_mask, dtype=bool)
        
#         # 전체 경계 요소 수: predicted boundary와 GT boundary의 총합
#         total_boundary = np.count_nonzero(pred_boundary) + np.count_nonzero(gt_boundary)
#         if total_boundary == 0:
#             # 해당 클래스가 양쪽 segmentation 모두에 존재하지 않는 경우
#             nsd[c] = np.nan
#             continue
        
#         # GT boundary 쪽의 distance transform: GT boundary가 아닌 영역에 대해 계산 후, predicted boundary 위치의 거리 추출
#         if np.any(gt_boundary):
#             dt_gt = distance_transform_edt(~gt_boundary)
#             distances_pred = dt_gt[pred_boundary]
#         else:
#             distances_pred = np.array([])
        
#         # predicted boundary 쪽의 distance transform: predicted boundary가 아닌 영역에 대해 계산 후, GT boundary 위치의 거리 추출
#         if np.any(pred_boundary):
#             dt_pred = distance_transform_edt(~pred_boundary)
#             distances_gt = dt_pred[gt_boundary]
#         else:
#             distances_gt = np.array([])
        
#         # threshold 이하인 요소 수 계산 (여기서는 threshold 값이 1.0로 통일됨)
#         correct_pred = np.sum(distances_pred <= threshold) if distances_pred.size > 0 else 0
#         correct_gt   = np.sum(distances_gt <= threshold)   if distances_gt.size > 0 else 0
        
#         nsd[c] = (correct_pred + correct_gt) / total_boundary
#         mean_nsd = np.nanmean(nsd)
        
#     return mean_nsd

def calculate_dice_score(predicted, ground_truth):

    # DiceMetric은 torch.Tensor를 입력으로 받으므로, numpy 배열을 tensor로 변환
    predicted_tensor = torch.from_numpy(predicted).float()
    ground_truth_tensor = torch.from_numpy(ground_truth).long()
    predicted_tensor = predicted_tensor.unsqueeze(0).unsqueeze(0)
    ground_truth_tensor = ground_truth_tensor.unsqueeze(0).unsqueeze(0)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True, ignore_empty=True)
    dice_score = dice_metric(y_pred=predicted_tensor, y=ground_truth_tensor)

    return dice_score.item()

# def calculate_nsd_score(predicted, ground_truth, threshold=1.0):
#     nsd = SurfaceDiceMetric(include_background=False, reduction="mean", get_not_nans=False, threshold=threshold)
#     nsd(y_pred=predicted, y=ground_truth)
#     nsd.aggregate()
#     mean_nsd = nsd.mean()
#     return mean_nsd.item()


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