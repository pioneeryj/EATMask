import inspect
import sys
sys.path.insert(0, '/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain')
from nnunetv2.training.nnUNetTrainer.variants.pretrain.STUNet_head import STUNet
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

device=torch.device("cuda:1")
# 변동
enc_checkpoint_path_dir = '/home/yoonji/AnatoMask/Anatomask_results/Dataset601_Total/Pretraining/0218/anatomask/anatomask_head_latest.pt'
result_dir = "/home/yoonji/AnatoMask/Anatomask_results/Dataset601_Total/Pretraining/anatomask/1000epoch"
# 데이터 관련
dec_pretrained_model_dir = "/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrained_model/large_ep4k.model"  # pretrained weights 경로
dataset_json_path_dir = "/home/yoonji/AnatoMask/Anatomask_results/Dataset601_Total/Pretraining/0224/anatomask/dataset.json"
dataset_dir = "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset601_organs_TotalSegmentator/imagesTs"
label_dir = "/nas_homes/yoonji/medmask/nnUNet_raw/Dataset606_all_TotalSegmentator/labelsTs"
import torch
import os
from torch import nn
from nnunetv2.training.nnUNetTrainer.variants.pretrain.encoder3D import SparseEncoder
from nnunetv2.training.nnUNetTrainer.variants.pretrain.decoder3D import LightDecoder
from nnunetv2.training.nnUNetTrainer.variants.pretrain.STUNet_head import STUNet, Decoder
from nnunetv2.training.nnUNetTrainer.variants.pretrain.STUNet import Decoder_forward, Encoder_forward
import argparse

 # Segmentation decoder 추가
import json
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from monai.metrics import DiceMetric
from metric_evaluation import center_crop, pad_to_target_size, calculate_dice_score

def load_dataset_json(json_path):
    """
    dataset.json 로드 및 이미지 확장자, 라벨 매핑 가져오기
    """
    with open(json_path, 'r') as f:
        dataset_info = json.load(f)
    
    file_extension = dataset_info.get('file_ending', '.nii.gz')  # Default: .nii.gz
    label_mapping = dataset_info.get('labels', {})  # 라벨 매핑 가져오기

    return dataset_info, file_extension, label_mapping

class MedicalImageDataset(Dataset):
    """
    3D Medical Image Dataset - dataset.json 기반 파일 로딩 + 라벨 매핑
    """
    def __init__(self, data_folder, device):
        self.device = device

        # 데이터 폴더 내의 파일 가져오기
        self.image_paths = [
            os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".nii.gz")
        ]
        self.image_paths.sort()  # 파일 정렬
        self.target_size = (112,112,128)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = nib.load(image_path).get_fdata()  # (H, W, D) 형태

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W, D)

        cropped_image = self.center_crop(image, self.target_size)
        padded_image = self.pad_to_target_size(cropped_image, self.target_size)

        padded_image = padded_image.to(self.device)

        return padded_image, image_path
    
    def center_crop(self, image, target_size):
        """
        중앙 크롭 (image: (1, H, W, D), target_size: (112, 112, 128))
        """
        _, H, W, D = image.shape
        crop_H, crop_W, crop_D = target_size

        # 크롭 시작 위치 계산 (중앙 기준)
        start_H = max((H - crop_H) // 2, 0)
        start_W = max((W - crop_W) // 2, 0)
        start_D = max((D - crop_D) // 2, 0)

        # 크롭 끝 위치 계산
        end_H = min(start_H + crop_H, H)
        end_W = min(start_W + crop_W, W)
        end_D = min(start_D + crop_D, D)

        return image[:, start_H:end_H, start_W:end_W, start_D:end_D]

    def pad_to_target_size(self, image, target_size):
        """
        부족한 크기를 패딩 (image: (1, h, w, d), target_size: (112, 112, 128))
        """
        _, h, w, d = image.shape
        pad_h = max(target_size[0] - h, 0)
        pad_w = max(target_size[1] - w, 0)
        pad_d = max(target_size[2] - d, 0)

        # (앞쪽 패딩, 뒤쪽 패딩) 설정
        pad_H = (pad_h // 2, pad_h - pad_h // 2)
        pad_W = (pad_w // 2, pad_w - pad_w // 2)
        pad_D = (pad_d // 2, pad_d - pad_d // 2)

        return F.pad(image, (pad_D[0], pad_D[1], pad_W[0], pad_W[1], pad_H[0], pad_H[1]), mode='constant', value=0)


class SegmentationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        features, skips = self.encoder(x)  # Pretrained Encoder 활용
        output = self.decoder(features, skips)  # Segmentation Decoder 적용
        return output

def load_pretrained_encoder(checkpoint_path, device):
    """
    MIM으로 학습된 encoder 로드
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Encoder 구조 정의 (훈련 때와 동일해야 함)
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
    conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

    # head = STUNet(1, 1, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], 
    #               pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
    #               enable_deep_supervision=True).to(device)

    # input_size = (112, 112, 128)
    # encoder = SparseEncoder(head, input_size=input_size, sbn=False).to(device)
    encoder = Encoder_forward(1, 1, depth=[1, 1, 1, 1, 1, 1], dims=[64, 128, 256, 512, 1024, 1024], 
                  pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                  enable_deep_supervision=True).to(device)

    # Checkpoint 로드
    state_dict = checkpoint['network_weights'] 
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(new_state_dict, strict=False)

    return encoder

def load_segmentation_decoder(checkpoint_path, device, encoder:List):
    """
    Segmentation decoder 로드
    """
    pretrained_weights = torch.load(checkpoint_path, map_location=device)
    decoder_state_dict = {k: v for k, v in pretrained_weights['state_dict'].items() 
                          if k.startswith("conv_blocks_localization") or k.startswith("upsample_layers") or k.startswith("seg_outputs")}
    
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
    conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    dims=[64, 128, 256, 512, 1024, 1024]
    num_classes = 105
    
    decoder = Decoder_forward(dims = dims, num_classes = num_classes, pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
                              depth =[1, 1, 1, 1, 1, 1]).to(device)
    decoder.load_state_dict(decoder_state_dict, strict=False)

    return decoder

def initialize_segmentation_model(enc_checkpoint, dec_checkpoint, device):
    """
    MIM으로 학습된 Encoder + Segmentation Decoder 로드
    """
    encoder = load_pretrained_encoder(enc_checkpoint, device)
    decoder = load_segmentation_decoder(dec_checkpoint, device, encoder)
    model = SegmentationModel(encoder, decoder).to(device)
    return model

def inference_segmentation_model(model, data_folder, dataset_json_path, output_folder, batch_size=1, num_workers=4):
    """
    DataLoader를 사용하여 Inference 수행 + 라벨 매핑 적용
    """
    
    dataset = MedicalImageDataset(data_folder, device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataset_info, _, label_mapping = load_dataset_json(dataset_json_path)
    label_mapping_inv = {v: k for k, v in label_mapping.items()}  # {숫자: 장기명} 형태로 변환

    model.eval()  # 모델을 평가 모드로 설정
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, image_paths) in enumerate(data_loader):
            outputs = model(inputs)  # Inference 수행
            outputs = torch.argmax(outputs[0], dim=1).cpu().numpy()  # deep supervision 첫번째 요소 tensor 선택

            for i, img_path in enumerate(image_paths):
                filename = os.path.basename(img_path).replace(".nii.gz", "_pred.nii.gz")  # 파일명 변경
                output_path = os.path.join(output_folder, filename)

                # 기존 이미지의 affine & header 유지
                original_img = nib.load(img_path)
                pred_img = nib.Nifti1Image(outputs[i], affine=original_img.affine, header=original_img.header)

                nib.save(pred_img, output_path)
                print(f"[{batch_idx + 1}/{len(data_loader)}] Saved: {output_path}")

def inference_and_evaluate(model, data_folder, dataset_json_path, label_dir, batch_size=1, num_workers=4):
    """
    DataLoader를 사용하여 Inference 수행 + Ground Truth와 비교하여 Dice Score 계산
    """
    
    dataset = MedicalImageDataset(data_folder, device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataset_info, _, label_mapping = load_dataset_json(dataset_json_path)
    label_mapping_inv = {v: k for k, v in label_mapping.items()}  # {숫자: 장기명} 형태로 변환

    model.eval()  # 모델을 평가 모드로 설정

    target_size = (112, 112, 128)

    dice_scores = []

    with torch.no_grad():
        for batch_idx, (inputs, image_paths) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)  # Inference 수행
            outputs = torch.argmax(outputs[0], dim=1).cpu().numpy()  # deep supervision 첫번째 요소 tensor 선택

            for i, img_path in enumerate(image_paths):
                filename = os.path.basename(img_path).replace("_0000.nii.gz", ".nii.gz")  # 원본 파일명
                label_path = os.path.join(label_dir, filename)  # 정답(GT) 이미지 경로
                
                if not os.path.exists(label_path):
                    print(f"Warning: {label_path} not found. Skipping.")
                    continue

                # GT 이미지 불러오기 (numpy array, shape: (H, W, D))
                gt_img = nib.load(label_path).get_fdata()

                # 정답(GT) 이미지 크롭 및 패딩 적용
                cropped_gt = center_crop(gt_img, target_size)
                padded_gt = pad_to_target_size(torch.from_numpy(cropped_gt), target_size).numpy()

                # Dice Score 계산
                dice_score = calculate_dice_score(outputs[i], padded_gt)
                dice_scores.append(dice_score)
                print(f"[{batch_idx + 1}/{len(data_loader)}] {filename}: Dice = {dice_score:.4f}")

    # 전체 평균 Dice Score 출력
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"\n🔥 Average Dice Score: {avg_dice:.4f}")
    return avg_dice    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="for path")
    
    parser.add_argument('--enc', help="enc_checkpoint_path", default=enc_checkpoint_path_dir)
    parser.add_argument('--dec', default=dec_pretrained_model_dir)
    parser.add_argument('--json', default=dataset_json_path_dir)
    parser.add_argument('--dataset', default=dataset_dir)
    parser.add_argument('--output', default=result_dir)
    parser.add_argument('--label', default = label_dir)
    
    args = parser.parse_args()
    
    enc_checkpoint_path = args.enc
    dec_pretrained_path = args.dec
    dataset_json_path = args.json
    dataset_folder = args.dataset
    out_folder = args.output
    label_dir = args.label

    device = torch.device("cuda:1")
    print(f"load encoder weight from : {enc_checkpoint_path}")
    segmentation_model = initialize_segmentation_model(enc_checkpoint_path, dec_pretrained_path, device)
    segmentation_model.to(device)
    # inference_segmentation_model(segmentation_model, dataset_folder, dataset_json_path, out_folder)
    inference_and_evaluate(segmentation_model, dataset_folder, dataset_json_path, label_dir)


