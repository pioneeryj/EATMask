
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import argparse
from time import time, sleep
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, '/home/yoonji/AnatoMask')

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import (
    load_json, maybe_mkdir_p, join
)
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import (
    RemoveLabelTransform, RenameTransform, NumpyToTensor
)
from nnunetv2.training.lr_scheduler.LinearWarmupCosine import LinearWarmupCosineAnnealingLR
from nnunetv2.training.nnUNetTrainer.STUNetVarianceTrainer import (
    VarianceHead, CombinedSegmentationVarianceLoss
)
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class ConvBlock3D(nn.Module):
    """3D Conv Block: Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ============================================================================
# NEW ARCHITECTURE: Shared Encoder + Separate Enhanced Decoders
# ============================================================================

class SimpleUNet3DEncoder(nn.Module):
    """
    Shared Encoder for both Mean and Variance prediction
    Extracts features from input image
    """
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Encoder level 1
        self.enc1 = nn.Sequential(
            ConvBlock3D(in_channels, base_channels),
            ConvBlock3D(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool3d(2)

        # Encoder level 2
        self.enc2 = nn.Sequential(
            ConvBlock3D(base_channels, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool3d(2)

        # Encoder level 3
        self.enc3 = nn.Sequential(
            ConvBlock3D(base_channels * 2, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4)
        )
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock3D(base_channels * 4, base_channels * 8),
            ConvBlock3D(base_channels * 8, base_channels * 8)
        )

    def forward(self, x):
        # Store skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(x)

        # Return bottleneck and skip connections
        return bottleneck, (enc1, enc2, enc3)


class SimpleUNet3DDecoder(nn.Module):
    """
    Decoder for Mean (Segmentation) prediction
    Same architecture as original SimpleUNet3D decoder
    """
    def __init__(self, num_classes=105, base_channels=32):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Decoder level 3
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock3D(base_channels * 8, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4)
        )

        # Decoder level 2
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock3D(base_channels * 4, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2)
        )

        # Decoder level 1
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock3D(base_channels * 2, base_channels),
            ConvBlock3D(base_channels, base_channels)
        )

        # Output
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def _match_size(self, x, target):
        """Pad or crop x to match target's spatial dimensions"""
        if x.shape[2:] != target.shape[2:]:
            # Interpolate to match target size
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x

    def forward(self, bottleneck, skip_connections):
        enc1, enc2, enc3 = skip_connections

        # Decoder with skip connections
        x = self.upconv3(bottleneck)
        x = self._match_size(x, enc3)  # Match size before concat
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = self._match_size(x, enc2)  # Match size before concat
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = self._match_size(x, enc1)  # Match size before concat
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        # Output
        x = self.final_conv(x)
        return x


class EnhancedVarianceDecoder(nn.Module):
    """
    Enhanced Decoder for Variance (Log-Variance) prediction
    """
    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = base_channels

        # Decoder level 3 - More capacity than mean decoder
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock3D(base_channels * 8, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4)  # Extra layer for variance
        )

        # Decoder level 2
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock3D(base_channels * 4, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2)  # Extra layer for variance
        )

        # Decoder level 1
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock3D(base_channels * 2, base_channels),
            ConvBlock3D(base_channels, base_channels),
            ConvBlock3D(base_channels, base_channels)  # Extra layer for variance
        )

        # Enhanced variance head (5 layers instead of 3)
        # This processes the decoded features to predict log-variance
        self.variance_head = nn.Sequential(
            nn.Conv3d(base_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),  # Extra layer
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),   # Extra layer
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=1)                # Final 1x1 conv
        )

    def _match_size(self, x, target):
        """Pad or crop x to match target's spatial dimensions"""
        if x.shape[2:] != target.shape[2:]:
            # Interpolate to match target size
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x

    def forward(self, bottleneck, skip_connections):
        enc1, enc2, enc3 = skip_connections

        # Decoder with skip connections (same as mean decoder but with extra layers)
        x = self.upconv3(bottleneck)
        x = self._match_size(x, enc3)  # Match size before concat
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = self._match_size(x, enc2)  # Match size before concat
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = self._match_size(x, enc1)  # Match size before concat
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        # Enhanced variance head
        log_var = self.variance_head(x)
        return log_var


# ============================================================================
# OLD ARCHITECTURE (Original Implementation - Kept for reference)
# ============================================================================
# class SimpleUNet3D(nn.Module):
#     """
#     간단한 3D U-Net (STUNet보다 훨씬 적은 parameter)
#     - 4개의 depth level (STUNet의 6개에서 감소)
#     - 더 적은 channel 수
#     - 대략 2-3M parameters (STUNet은 100M+)
#     """
#     def __init__(self, in_channels=1, num_classes=105, base_channels=32):
#         super().__init__()
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.base_channels = base_channels
#
#         # Encoder
#         self.enc1 = nn.Sequential(
#             ConvBlock3D(in_channels, base_channels),
#             ConvBlock3D(base_channels, base_channels)
#         )
#         self.pool1 = nn.MaxPool3d(2)
#
#         self.enc2 = nn.Sequential(
#             ConvBlock3D(base_channels, base_channels * 2),
#             ConvBlock3D(base_channels * 2, base_channels * 2)
#         )
#         self.pool2 = nn.MaxPool3d(2)
#
#         self.enc3 = nn.Sequential(
#             ConvBlock3D(base_channels * 2, base_channels * 4),
#             ConvBlock3D(base_channels * 4, base_channels * 4)
#         )
#         self.pool3 = nn.MaxPool3d(2)
#
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             ConvBlock3D(base_channels * 4, base_channels * 8),
#             ConvBlock3D(base_channels * 8, base_channels * 8)
#         )
#
#         # Decoder
#         self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
#         self.dec3 = nn.Sequential(
#             ConvBlock3D(base_channels * 8, base_channels * 4),
#             ConvBlock3D(base_channels * 4, base_channels * 4)
#         )
#
#         self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
#         self.dec2 = nn.Sequential(
#             ConvBlock3D(base_channels * 4, base_channels * 2),
#             ConvBlock3D(base_channels * 2, base_channels * 2)
#         )
#
#         self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
#         self.dec1 = nn.Sequential(
#             ConvBlock3D(base_channels * 2, base_channels),
#             ConvBlock3D(base_channels, base_channels)
#         )
#
#         # Output
#         self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)
#
#     def forward(self, x):
#         # Encoder with skip connections
#         enc1 = self.enc1(x)
#         x = self.pool1(enc1)
#
#         enc2 = self.enc2(x)
#         x = self.pool2(enc2)
#
#         enc3 = self.enc3(x)
#         x = self.pool3(enc3)
#
#         # Bottleneck
#         x = self.bottleneck(x)
#
#         # Decoder with skip connections
#         x = self.upconv3(x)
#         x = torch.cat([x, enc3], dim=1)
#         x = self.dec3(x)
#
#         x = self.upconv2(x)
#         x = torch.cat([x, enc2], dim=1)
#         x = self.dec2(x)
#
#         x = self.upconv1(x)
#         x = torch.cat([x, enc1], dim=1)
#         x = self.dec1(x)
#
#         # Output
#         x = self.final_conv(x)
#         return x


# OLD SimpleUNetWithVarianceHead (Original - Sequential Architecture)
# class SimpleUNetWithVarianceHead(nn.Module):
#     """SimpleUNet + Variance Head for uncertainty estimation"""
#     def __init__(self, base_model, num_classes=105):
#         super().__init__()
#         self.network = base_model
#         self.variance_head = VarianceHead(num_classes=num_classes)
#
#     def forward(self, x):
#         # Get segmentation logits
#         seg_logits = self.network(x)
#
#         # Get variance
#         log_var = self.variance_head(seg_logits)
#
#         return {
#             'seg_logits': seg_logits,
#             'log_var': log_var
#         }


# ============================================================================
# NEW SimpleUNetWithVarianceHead
# ============================================================================
class SimpleUNetWithVarianceHead(nn.Module):
    """
    NEW ARCHITECTURE: Shared Encoder + Separate Decoders for Mean and Variance

    Benefits:
    - Variance decoder directly accesses encoder features (no information bottleneck)
    - Mean and variance are predicted in parallel (not sequentially)
    - Enhanced capacity for complex uncertainty patterns
    """
    def __init__(self, in_channels=1, num_classes=105, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Shared encoder for both mean and variance
        self.encoder = SimpleUNet3DEncoder(in_channels=in_channels, base_channels=base_channels)

        # Separate decoder for mean (segmentation)
        self.decoder_mean = SimpleUNet3DDecoder(num_classes=num_classes, base_channels=base_channels)

        # Enhanced separate decoder for variance
        self.decoder_var = EnhancedVarianceDecoder(base_channels=base_channels)

    def forward(self, x):
        """
        Forward pass with shared encoder and separate decoders

        Args:
            x: Input image [B, C, H, W, D]

        Returns:
            dict with 'seg_logits' and 'log_var'
        """
        # Shared encoding
        bottleneck, skip_connections = self.encoder(x)

        # Parallel decoding for mean and variance
        seg_logits = self.decoder_mean(bottleneck, skip_connections)
        log_var = self.decoder_var(bottleneck, skip_connections)

        return {
            'seg_logits': seg_logits,
            'log_var': log_var
        }


def get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                            order_resampling_data=3, order_resampling_seg=1,
                            use_mask_for_norm=None):
    """데이터 증강 파이프라인 (pretrain_MedMask.py 기반)"""
    tr_transforms = []

    # Spatial Transform: 회전, 스케일링, 엘라스틱 변형
    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True,
        angle_x=rotation_for_DA['x'],
        angle_y=rotation_for_DA['y'],
        angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=-1, order_seg=order_resampling_seg,
        random_crop=False,
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False
    ))

    # Mirror Transform
    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    # Mask Transform for normalization (선택적)
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
        tr_transforms.append(MaskTransform(
            [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
            mask_idx_in_seg=0, set_outside_to=0))

    # Remove background label
    tr_transforms.append(RemoveLabelTransform(-1, 0))

    # Rename seg to target
    tr_transforms.append(RenameTransform('seg', 'target', True))

    # Deep Supervision downsampling (only if enabled)
    # For this training, deep_supervision_scales is None, so skip this
    if deep_supervision_scales is not None and len(deep_supervision_scales) > 0:
        from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import DownsampleSegForDSTransform2
        tr_transforms.append(DownsampleSegForDSTransform2(
            deep_supervision_scales, 0, input_key='target', output_key='target'))

    # Convert to Tensor
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    return Compose(tr_transforms)


def get_validation_transforms(deep_supervision_scales, use_mask_for_norm=None):
    """검증 데이터 파이프라인 (pretrain_MedMask.py 기반)"""
    val_transforms = []

    # Remove background label
    val_transforms.append(RemoveLabelTransform(-1, 0))

    # Mask Transform for normalization (선택적)
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
        val_transforms.append(MaskTransform(
            [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
            mask_idx_in_seg=0, set_outside_to=0))

    # Rename seg to target
    val_transforms.append(RenameTransform('seg', 'target', True))

    # Deep Supervision downsampling (only if enabled)
    # For this training, deep_supervision_scales is None, so skip this
    if deep_supervision_scales is not None and len(deep_supervision_scales) > 0:
        from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import DownsampleSegForDSTransform2
        val_transforms.append(DownsampleSegForDSTransform2(
            deep_supervision_scales, 0, input_key='target', output_key='target'))

    # Convert to Tensor
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    return Compose(val_transforms)


def main():
    """메인 함수 with error handling for background execution"""
    try:
        _main_impl()
    except Exception as e:
        print(f"\n{'='*70}", flush=True)
        print(f"FATAL ERROR: {str(e)}", flush=True)
        print(f"{'='*70}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _main_impl():
    """메인 함수 implementation"""

    # ===== Arguments =====
    parser = argparse.ArgumentParser(description='Train SimpleUNet with Aleatoric Uncertainty')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup epochs')
    parser.add_argument('--output', type=str, default=None, help='Output folder')
    parser.add_argument('--model_name', type=str, default='simpleunet_aleatoric', help='Model name')

    args = parser.parse_args()

    # ===== 디바이스 설정 =====
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== 경로 설정 =====
    preprocessed_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/nnUNetPlans_3d_fullres'
    dataset_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/dataset.json'
    plans_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/nnUNetPlans.json'
    splits_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/splits_final.json'

    if args.output is None:
        base_folder = '/home/yoonji/AnatoMask/Anatomask_results/aleatoric_uncertainty'
        output_folder = join(base_folder, args.model_name, f'fold{args.fold}')
    else:
        output_folder = args.output

    maybe_mkdir_p(output_folder)

    # ===== 데이터 로드 =====
    print("\n📚 Loading data...")

    dataset_json = load_json(dataset_json_path)
    plans = load_json(plans_json_path)
    splits = load_json(splits_path)

    all_keys = splits[args.fold]['train']
    tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

    dataset_tr = nnUNetDataset(preprocessed_path, tr_keys, folder_with_segs_from_previous_stage=None,
                               num_images_properties_loading_threshold=0)
    dataset_val = nnUNetDataset(preprocessed_path, val_keys, folder_with_segs_from_previous_stage=None,
                                num_images_properties_loading_threshold=0)

    print(f"✓ Training samples: {len(dataset_tr)}")
    print(f"✓ Validation samples: {len(dataset_val)}")

    # ===== 플랜 및 설정 =====
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    label_manager = plans_manager.get_label_manager(dataset_json)

    patch_size = configuration_manager.patch_size
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]

    # Deep supervision disabled - only use final resolution
    deep_supervision_scales = None

    mirror_axes = (0, 1, 2)

    rotation_for_DA = {
        'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    }

    initial_patch_size = get_patch_size(patch_size, *rotation_for_DA.values(), (0.85, 1.25))

    batch_size = args.batch_size
    oversample_foreground_percent = 0.33

    # ===== 데이터 로더 =====
    print("\n⚙️  Setting up data loaders...")

    # Training transforms (pretrain_MedMask.py 기반)
    # deep_supervision_scales는 None (단일 해상도만 사용)
    tr_transforms = get_training_transforms(
        patch_size,
        rotation_for_DA,
        None,  # deep_supervision_scales = None (no downsampling)
        mirror_axes,
        order_resampling_data=3,
        order_resampling_seg=1,
        use_mask_for_norm=configuration_manager.use_mask_for_norm
    )

    # Validation transforms (pretrain_MedMask.py 기반)
    # deep_supervision_scales가 None이므로 downsampling 없음
    val_transforms = get_validation_transforms(
        None,  # explicitly pass None to avoid deep supervision
        use_mask_for_norm=configuration_manager.use_mask_for_norm
    )

    # Data loader (3D patch-based, no transforms yet)
    dl_tr = nnUNetDataLoader3D(
        dataset_tr,
        batch_size,
        initial_patch_size,
        patch_size,
        label_manager,
        oversample_foreground_percent=oversample_foreground_percent,
        sampling_probabilities=None,
        pad_sides=None
    )

    # Data loader for validation
    dl_val = nnUNetDataLoader3D(
        dataset_val,
        batch_size,
        initial_patch_size,
        patch_size,
        label_manager,
        oversample_foreground_percent=0,  # No oversampling for validation
        sampling_probabilities=None,
        pad_sides=None
    )

    iters_train = len(dataset_tr) // batch_size
    iters_val = len(dataset_val) // batch_size

    print(f"✓ Training iterations per epoch: {iters_train}")
    print(f"✓ Validation iterations per epoch: {iters_val}")

    # Apply transforms using LimitedLenWrapper (pretrain_MedMask.py 기반)
    # This applies augmentation in a multithreaded manner
    allowed_num_processes = get_allowed_n_proc_DA()

    print(f"\n⚙️  Setting up multithreaded augmenters (workers: {allowed_num_processes})...")

    # Training data with augmentation
    mt_gen_train = LimitedLenWrapper(
        iters_train,
        data_loader=dl_tr,
        transform=tr_transforms,
        num_processes=allowed_num_processes,
        num_cached=6,
        seeds=None,
        pin_memory=True,
        wait_time=0.02
    )

    # Validation data with augmentation (no spatial augmentation, only preprocessing)
    mt_gen_val = LimitedLenWrapper(
        iters_val,
        data_loader=dl_val,
        transform=val_transforms,
        num_processes=allowed_num_processes,
        num_cached=2,
        seeds=None,
        pin_memory=True,
        wait_time=0.02
    )

    print(f"✓ Data augmentation pipeline ready")

    # ===== 모델 구축 =====
    print("\n🧠 Building SimpleUNet with Variance Head (NEW: Shared Encoder + Separate Decoders)...")

    # OLD: Sequential architecture (commented out)
    # unet = SimpleUNet3D(in_channels=1, num_classes=105, base_channels=32)
    # model = SimpleUNetWithVarianceHead(unet, num_classes=105)

    # NEW: Shared encoder + separate enhanced decoders
    model = SimpleUNetWithVarianceHead(in_channels=1, num_classes=105, base_channels=32)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")

    # ===== 손실함수, 옵티마이저, 스케줄러 =====
    print("\n⚙️  Setting up training...")

    # Segmentation Loss (Dice) + Variance Loss
    loss_fn = CombinedSegmentationVarianceLoss(num_classes=105, seg_weight=1.0, var_weight=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, args.warmup, args.epochs, 1e-6)

    # ===== 로거 =====
    log_file = join(output_folder, f'training_log_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.txt')

    def print_to_log_file(*args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp))
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    # ===== 학습 루프 =====
    print_to_log_file("\n" + "="*70)
    print_to_log_file("STARTING SIMPLEUNET ALEATORIC UNCERTAINTY TRAINING")
    print_to_log_file("="*70)
    print_to_log_file(f"Model parameters: {num_params:,}")
    print_to_log_file(f"Batch size: {batch_size}")
    print_to_log_file(f"Learning rate: {args.lr}")
    print_to_log_file(f"Warmup epochs: {args.warmup}")
    print_to_log_file(f"Total epochs: {args.epochs}")

    it = 0
    epoch_loss = []
    epoch_seg_loss = []
    epoch_var_loss = []
    epoch_val_loss = []
    epoch_val_seg_loss = []
    epoch_val_var_loss = []

    for epoch in range(args.epochs):
        model.train()
        per_loss = 0.0
        per_seg_loss = 0.0
        per_var_loss = 0.0

        print_to_log_file('')
        print_to_log_file(f'Epoch {epoch}')
        print_to_log_file(f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")

        pbar = tqdm(range(iters_train), desc=f"Epoch {epoch}")

        for train_iter in pbar:
            try:
                # LimitedLenWrapper에서 augmented batch를 받음
                # transforms가 이미 적용되었으므로 tensor 형태임
                batch = next(mt_gen_train)

                # Data and target (transforms에서 'seg' -> 'target'으로 변환됨)
                inp = batch['data']
                target = batch['target']

                # Handle list case by padding to common size (same as validation)
                def pad_tensors_to_common_size(tensor_list):
                    """Pad list of tensors to have common size in all dimensions"""
                    if not isinstance(tensor_list, (list, tuple)):
                        return tensor_list

                    # Find max size in each dimension
                    ndim = tensor_list[0].ndim
                    max_shape = list(tensor_list[0].shape)

                    for t in tensor_list[1:]:
                        for d in range(ndim):
                            max_shape[d] = max(max_shape[d], t.shape[d])

                    # Pad each tensor to max_shape
                    padded_tensors = []
                    for t in tensor_list:
                        if list(t.shape) != max_shape:
                            # Calculate padding for each dimension (pad from right/bottom)
                            padding = []
                            for d in range(ndim - 1, -1, -1):  # reverse order for F.pad
                                pad_size = max_shape[d] - t.shape[d]
                                padding.extend([0, pad_size])

                            t_padded = F.pad(t, padding, mode='constant', value=0)
                            padded_tensors.append(t_padded)
                        else:
                            padded_tensors.append(t)

                    # Stack into single tensor
                    return torch.stack(padded_tensors, dim=0)

                # DEBUG: Print target type and shape (only for first batch)
                if epoch == 0 and train_iter == 0:
                    print(f"[DEBUG Train] inp type: {type(inp)}", flush=True)
                    print(f"[DEBUG Train] target type: {type(target)}", flush=True)
                    if isinstance(target, (list, tuple)):
                        print(f"[DEBUG Train] Target is list/tuple with {len(target)} elements", flush=True)
                        for i, t in enumerate(target):
                            print(f"[DEBUG Train]   [{i}] shape: {t.shape}, dtype: {t.dtype}", flush=True)
                    else:
                        print(f"[DEBUG Train] Target shape: {target.shape}, dtype: {target.dtype}", flush=True)

                # Pad if list (handles both deep supervision and size mismatches)
                if isinstance(inp, (list, tuple)):
                    if epoch == 0 and train_iter == 0:
                        print(f"[DEBUG Train] Padding inp list to common size", flush=True)
                    inp = pad_tensors_to_common_size(inp)

                if isinstance(target, (list, tuple)):
                    if epoch == 0 and train_iter == 0:
                        print(f"[DEBUG Train] Padding target list to common size", flush=True)
                    # For deep supervision, use last element only
                    target = target[-1] if len(target) > 0 else target[0]

                # Move to device
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Forward pass
                output = model(inp)

                # Resize target to match model output spatial dimensions
                pred_size = output['seg_logits'].shape[2:]  # [H, W, D]

                # Ensure target is 4D [B, H, W, D]
                if target.ndim == 5:  # [B, 1, H, W, D]
                    target = target.squeeze(1)  # [B, H, W, D]

                # Interpolate if spatial dims don't match
                if target.shape[1:] != pred_size:
                    # target is [B, H, W, D], need to add channel for interpolate
                    target = F.interpolate(
                        target.unsqueeze(1).float(),  # [B, 1, H, W, D]
                        size=pred_size,
                        mode='trilinear',
                        align_corners=False
                    ).long().squeeze(1)  # [B, H', W', D']

                # Loss 계산 (output: {'seg_logits': ..., 'log_var': ...})
                loss_dict = loss_fn(output['seg_logits'], target, output['log_var'])
            except Exception as e:
                print_to_log_file(f"Error during training iteration: {str(e)}")
                print(f"Error during training: {str(e)}", flush=True)
                continue

            # Backward
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()

            total_loss = loss_dict['total'].item()
            seg_loss = loss_dict['seg'].item()
            var_loss = loss_dict['var'].item()

            per_loss += total_loss
            per_seg_loss += seg_loss
            per_var_loss += var_loss

            # Extract log_var for logging
            log_var_mean = output['log_var'].mean().item()

            pbar.set_postfix({
                'loss': f'{total_loss:.4f}',
                'seg_loss': f'{seg_loss:.4f}',
                'var_loss': f'{var_loss:.4f}',
                'log_var': f'{log_var_mean:.4f}'
            })

            it += 1

        scheduler.step()

        avg_epoch_loss = per_loss / iters_train
        avg_seg_loss = per_seg_loss / iters_train
        avg_var_loss = per_var_loss / iters_train

        epoch_loss.append(avg_epoch_loss)
        epoch_seg_loss.append(avg_seg_loss)
        epoch_var_loss.append(avg_var_loss)

        print_to_log_file(f"Epoch {epoch} AVG Loss: {avg_epoch_loss:.6f} (Seg: {avg_seg_loss:.6f}, Var: {avg_var_loss:.6f})")

        # ===== Validation =====
        print(f"\n🔍 Validating epoch {epoch}...")
        model.eval()
        per_val_loss = 0.0
        per_val_seg_loss = 0.0
        per_val_var_loss = 0.0

        with torch.no_grad():
            for val_iter in range(iters_val):
                try:
                    batch = next(mt_gen_val)

                    inp = batch['data']
                    target = batch['target']

                    # Handle list case by padding to common size
                    def pad_tensors_to_common_size(tensor_list):
                        """Pad list of tensors to have common size in all dimensions"""
                        if not isinstance(tensor_list, (list, tuple)):
                            return tensor_list

                        # Find max size in each dimension (excluding batch dim)
                        ndim = tensor_list[0].ndim
                        max_shape = list(tensor_list[0].shape)

                        for t in tensor_list[1:]:
                            for d in range(ndim):
                                max_shape[d] = max(max_shape[d], t.shape[d])

                        # Pad each tensor to max_shape
                        padded_tensors = []
                        for t in tensor_list:
                            if list(t.shape) != max_shape:
                                # Calculate padding for each dimension (pad from right/bottom)
                                padding = []
                                for d in range(ndim - 1, -1, -1):  # reverse order for F.pad
                                    pad_size = max_shape[d] - t.shape[d]
                                    padding.extend([0, pad_size])

                                t_padded = F.pad(t, padding, mode='constant', value=0)
                                padded_tensors.append(t_padded)
                            else:
                                padded_tensors.append(t)

                        # Stack into single tensor
                        return torch.stack(padded_tensors, dim=0)

                    # DEBUG: Print data info
                    if val_iter == 0:
                        print(f"[DEBUG Val] inp type: {type(inp)}", flush=True)
                        print(f"[DEBUG Val] target type: {type(target)}", flush=True)
                        if isinstance(target, (list, tuple)):
                            print(f"[DEBUG Val] target is list with {len(target)} elements", flush=True)
                            for i, t in enumerate(target):
                                print(f"[DEBUG Val]   target[{i}] shape: {t.shape}", flush=True)
                        else:
                            print(f"[DEBUG Val] target shape: {target.shape}", flush=True)

                    # Pad if list (handles both deep supervision and size mismatches)
                    if isinstance(inp, (list, tuple)):
                        if val_iter == 0:
                            print(f"[DEBUG Val] Padding inp list to common size", flush=True)
                        inp = pad_tensors_to_common_size(inp)

                    if isinstance(target, (list, tuple)):
                        if val_iter == 0:
                            print(f"[DEBUG Val] Padding target list to common size", flush=True)
                        # For deep supervision, use last element only
                        target = target[-1] if len(target) > 0 else target[0]

                    inp = inp.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    # Forward pass
                    output = model(inp)

                    # Resize target to match model output spatial dimensions
                    pred_size = output['seg_logits'].shape[2:]  # [H, W, D]

                    # Ensure target is 4D [B, H, W, D]
                    if target.ndim == 5:  # [B, 1, H, W, D]
                        target = target.squeeze(1)  # [B, H, W, D]

                    # Interpolate if spatial dims don't match
                    if target.shape[1:] != pred_size:
                        # target is [B, H, W, D], need to add channel for interpolate
                        target = F.interpolate(
                            target.unsqueeze(1).float(),  # [B, 1, H, W, D]
                            size=pred_size,
                            mode='trilinear',
                            align_corners=False
                        ).long().squeeze(1)  # [B, H', W', D']

                    # Loss 계산
                    loss_dict = loss_fn(output['seg_logits'], target, output['log_var'])

                    per_val_loss += loss_dict['total'].item()
                    per_val_seg_loss += loss_dict['seg'].item()
                    per_val_var_loss += loss_dict['var'].item()
                except Exception as e:
                    print_to_log_file(f"Error during validation iteration: {str(e)}")
                    print(f"Error during validation: {str(e)}", flush=True)
                    continue

        avg_val_loss = per_val_loss / iters_val
        avg_val_seg_loss = per_val_seg_loss / iters_val
        avg_val_var_loss = per_val_var_loss / iters_val

        print_to_log_file(f"Epoch {epoch} VAL Loss: {avg_val_loss:.6f} (Seg: {avg_val_seg_loss:.6f}, Var: {avg_val_var_loss:.6f})")

        # Save validation loss history
        epoch_val_loss.append(avg_val_loss)
        epoch_val_seg_loss.append(avg_val_seg_loss)
        epoch_val_var_loss.append(avg_val_var_loss)

        # 체크포인트 저장
        checkpoint = {
            'network_weights': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'seg_loss': epoch_seg_loss,
            'var_loss': epoch_var_loss,
            'val_loss': epoch_val_loss,
            'val_seg_loss': epoch_val_seg_loss,
            'val_var_loss': epoch_val_var_loss,
        }

        torch.save(checkpoint, join(output_folder, f'{args.model_name}_epoch_{epoch:03d}.pt'))

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save(checkpoint, join(output_folder, f'{args.model_name}_best.pt'))

    print_to_log_file("\n" + "="*70)
    print_to_log_file("✅ Training completed!")
    print_to_log_file("="*70)

    print(f"\n✅ Model saved to {output_folder}")


if __name__ == '__main__':
    main()
