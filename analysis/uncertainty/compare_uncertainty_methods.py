"""
Variance 기반 vs Entropy 기반 Aleatoric Uncertainty 비교

두 가지 방식으로 측정한 불확실성 맵을 시각화하고 비교합니다:
1. Variance-based: exp(log_var) 사용 (학습된 VarianceHead에서)
2. Entropy-based: -p*log(p) + (1-p)*log(1-p) 사용 (확률의 조건부 엔트로피)

둘 다 학습된 SimpleUNet 기반입니다.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import argparse

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
    """3D Conv Block: Conv -> BatchNorm -> ReLU -> Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_p=0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_p)  # MC Dropout을 위한 Dropout 추가

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout 적용
        return x


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

def mc_dropout(model, inp: torch.Tensor, T: int = 10):
    """
    Monte Carlo Dropout로 T회 로짓을 수집.
    메모리 효율적인 버전: 모든 계산을 CPU에서 수행

    Returns: List[torch.Tensor] of seg_logits (각 shape [B, C, H, W, D]) on CPU
    """
    logits_list = []
    was_training = model.training
    model.train()  # Dropout 활성

    print(f"  [MC Dropout] Running {T} forward passes...")
    print(f"  [MC Dropout] Model in training mode: {model.training}")

    # Dropout이 활성화되었는지 확인
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            dropout_count += 1
    print(f"  [MC Dropout] Found {dropout_count} dropout layers in model")

    with torch.no_grad():
        for i in range(T):
            # Forward pass
            out = model(inp)
            seg_logits = out['seg_logits'] if isinstance(out, dict) else out

            # 즉시 CPU로 이동 (GPU 메모리 최소화)
            logits_list.append(seg_logits.detach().cpu().clone())

            # GPU 메모리 즉시 정리
            del out, seg_logits
            torch.cuda.empty_cache()

            if (i + 1) % max(1, T // 5) == 0:
                print(f"  [MC Dropout] Progress: {i+1}/{T}")

    # 원래 모드로 복원
    if not was_training:
        model.eval()

    # Variance 확인 (디버깅)
    if len(logits_list) >= 2:
        logits_stack = torch.stack(logits_list[:2], dim=0)
        diff = (logits_stack[0] - logits_stack[1]).abs().mean().item()
        print(f"  [MC Dropout] Difference between first two samples: {diff:.6f}")
        if diff < 1e-6:
            print(f"  [WARNING] MC Dropout samples are identical! Dropout may not be working.")
            print(f"  [WARNING] This will result in zero epistemic uncertainty.")

    print(f"  [MC Dropout] Completed. Results stored on CPU.")
    return logits_list  # CPU에 유지

def calculate_softmax(logits):
    """
    logits: Tensor of shape (B, C, H, W)
    B: Batch size, C: Number of classes, H: Height, W: Width
    Returns: Softmax probabilities (B, C, H, W)
    """
    return F.softmax(logits, dim=1)

# 2. 픽셀 단위 조건부 엔트로피 계산
def calculate_conditional_entropy(probabilities):
    """
    probabilities: Tensor of shape (B, C, H, W)
    Returns: Conditional entropy map of shape (B, H, W)
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-8
    log_probs = torch.log(probabilities + epsilon)
    entropy = -torch.sum(probabilities * log_probs, dim=1)  # Sum over classes
    return entropy  # Shape: (B, H, W)

def calculate_epistemic_uncertainty(logits_list, device='cpu'):
    '''
    Epistemic Uncertainty: Variance of predictions across MC samples
    (AnatoMask.py 스타일, multi-class 버전)

    logits_list : list of len T, each element tensor [b,c,h,w,d] on CPU
    output : [b,1,h,w,d] on specified device
    '''
    print(f"  [Epistemic] Calculating epistemic uncertainty from {len(logits_list)} samples...")

    # Stack logits from MC samples
    logits_stack = torch.stack(logits_list, dim=0)  # [T, b, c, h, w, d]

    # Apply Softmax to convert logits to probabilities (multi-class)
    probs_stack = F.softmax(logits_stack, dim=2)  # [T, b, c, h, w, d], softmax over class dim

    # Variance of probabilities across MC samples (epistemic uncertainty)
    var_probs = probs_stack.var(dim=0)  # [b, c, h, w, d] - variance across T samples

    # Sum over classes to get total epistemic uncertainty per pixel
    # (AnatoMask uses squeeze for binary class where c=1, we sum for multi-class where c=105)
    epistemic_uncertainty = var_probs.sum(dim=1, keepdim=True)  # [b, 1, h, w, d]

    # 최종 결과만 GPU로 이동
    if device != 'cpu' and torch.cuda.is_available():
        epistemic_uncertainty = epistemic_uncertainty.to(device)

    print(f"  [Epistemic] Calculation complete.")
    print(f"  [Epistemic] Output shape: {epistemic_uncertainty.shape}")
    return epistemic_uncertainty


def calculate_aleatoric_uncertainty(logits_list, device='cpu'):
    '''
    Aleatoric Uncertainty: Average entropy across MC samples
    (AnatoMask.py 스타일, multi-class 버전)

    logits_list : list of len T, each element tensor [b,c,h,w,d] on CPU
    output : [b,1,h,w,d] on specified device
    '''
    print(f"  [Aleatoric] Calculating aleatoric uncertainty from {len(logits_list)} samples...")

    # Stack logits from MC samples
    logits_stack = torch.stack(logits_list, dim=0)  # [T, b, c, h, w, d]

    # Apply Softmax to convert logits to probabilities (multi-class)
    prob_samples = F.softmax(logits_stack, dim=2)  # [T, b, c, h, w, d]

    # Calculate entropy for each MC sample
    # Entropy = -sum(p * log(p)) over classes
    epsilon = 1e-8
    log_probs = torch.log(prob_samples + epsilon)
    entropy_samples = -torch.sum(prob_samples * log_probs, dim=2)  # [T, b, h, w, d], sum over class dim

    # Average entropy across MC samples
    aleatoric_uncertainty = entropy_samples.mean(dim=0)  # [b, h, w, d]

    # Add channel dimension
    aleatoric_uncertainty = aleatoric_uncertainty.unsqueeze(1)  # [b, 1, h, w, d]

    # 최종 결과만 GPU로 이동
    if device != 'cpu' and torch.cuda.is_available():
        aleatoric_uncertainty = aleatoric_uncertainty.to(device)

    print(f"  [Aleatoric] Calculation complete.")
    print(f"  [Aleatoric] Output shape: {aleatoric_uncertainty.shape}")
    return aleatoric_uncertainty


def visualize_uncertainty_comparison(original_img, seg_logits, var_uncertainty,
                                     aleatoric_uncertainty, epistemic_uncertainty,
                                     segmentation, slice_idx, save_path=None, sample_name="sample"):
    """
    모든 불확실성 맵 비교 시각화 (Original Image 포함)

    Args:
        original_img: [B, C, H, W, D] original input image
        seg_logits: [B, C, H, W, D]
        var_uncertainty: [B, 1, H, W, D] variance 기반 (learned)
        aleatoric_uncertainty: [B, 1, H, W, D] entropy 기반 (MC dropout)
        epistemic_uncertainty: [B, 1, H, W, D] epistemic (MC dropout)
        segmentation: [B, H, W, D]
        slice_idx: 시각화할 slice 인덱스
        save_path: 저장 경로
        sample_name: 샘플 이름
    """
    # 차원 정리 함수
    def squeeze_tensor(tensor):
        if tensor.ndim == 5:
            return tensor.squeeze(0).squeeze(0)
        elif tensor.ndim == 4:
            return tensor.squeeze(0)
        return tensor

    # 모든 텐서 정리
    original_img = squeeze_tensor(original_img)
    var_uncertainty = squeeze_tensor(var_uncertainty)
    aleatoric_uncertainty = squeeze_tensor(aleatoric_uncertainty)
    epistemic_uncertainty = squeeze_tensor(epistemic_uncertainty)
    segmentation = squeeze_tensor(segmentation)

    # CPU로 이동
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    original_img = to_numpy(original_img)
    var_uncertainty = to_numpy(var_uncertainty)
    aleatoric_uncertainty = to_numpy(aleatoric_uncertainty)
    epistemic_uncertainty = to_numpy(epistemic_uncertainty)
    segmentation = to_numpy(segmentation)

    # 중간 slice 추출
    mid_slice = var_uncertainty.shape[2] // 2 if slice_idx is None else slice_idx

    orig_slice = original_img[:, :, mid_slice]
    var_slice = var_uncertainty[:, :, mid_slice]
    aleatoric_slice = aleatoric_uncertainty[:, :, mid_slice]
    epistemic_slice = epistemic_uncertainty[:, :, mid_slice]
    seg_slice = segmentation[:, :, mid_slice]

    # Normalize for better visualization
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    var_slice_norm = normalize(var_slice)
    aleatoric_slice_norm = normalize(aleatoric_slice)
    epistemic_slice_norm = normalize(epistemic_slice)

    # 시각화: 1 row x 4 cols
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Col 1: Original Image
    im1 = axes[0].imshow(orig_slice, cmap='gray')
    axes[0].set_title(f'Original Image\n(Slice {mid_slice})', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Intensity', fraction=0.046, pad=0.04)

    # Col 2: Variance-based Uncertainty
    im2 = axes[1].imshow(var_slice_norm, cmap='hot')
    axes[1].set_title(f'Variance-based Uncertainty\n(Learned from Training)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Normalized', fraction=0.046, pad=0.04)

    # Col 3: Aleatoric Uncertainty
    im3 = axes[2].imshow(aleatoric_slice_norm, cmap='hot')
    axes[2].set_title(f'Aleatoric Uncertainty\n(Entropy from MC Dropout)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], label='Normalized', fraction=0.046, pad=0.04)

    # Col 4: Epistemic Uncertainty
    im4 = axes[3].imshow(epistemic_slice_norm, cmap='hot')
    axes[3].set_title(f'Epistemic Uncertainty\n(Variance from MC Dropout)', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], label='Normalized', fraction=0.046, pad=0.04)

    plt.suptitle(f'Uncertainty Analysis - {sample_name} (Slice: {mid_slice})',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 저장
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def print_statistics(var_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, label=""):
    """불확실성 맵의 통계정보 출력"""
    var_np = var_uncertainty.cpu().numpy()
    aleatoric_np = aleatoric_uncertainty.cpu().numpy()
    epistemic_np = epistemic_uncertainty.cpu().numpy()

    print(f"\n{label}")
    print(f"  Variance-based (Learned):  min={var_np.min():.4f}, max={var_np.max():.4f}, mean={var_np.mean():.4f}, std={var_np.std():.4f}")
    print(f"  Aleatoric (MC Entropy):    min={aleatoric_np.min():.4f}, max={aleatoric_np.max():.4f}, mean={aleatoric_np.mean():.4f}, std={aleatoric_np.std():.4f}")
    print(f"  Epistemic (MC Variance):   min={epistemic_np.min():.4f}, max={epistemic_np.max():.4f}, mean={epistemic_np.mean():.4f}, std={epistemic_np.std():.4f}")

    # Correlation 계산
    var_flat = var_np.flatten()
    aleatoric_flat = aleatoric_np.flatten()
    epistemic_flat = epistemic_np.flatten()

    corr_var_aleat = np.corrcoef(var_flat, aleatoric_flat)[0, 1]
    corr_var_epist = np.corrcoef(var_flat, epistemic_flat)[0, 1]
    corr_aleat_epist = np.corrcoef(aleatoric_flat, epistemic_flat)[0, 1]

    print(f"\n  Correlations:")
    print(f"    Variance ↔ Aleatoric:  {corr_var_aleat:.4f}")
    print(f"    Variance ↔ Epistemic:  {corr_var_epist:.4f}")
    print(f"    Aleatoric ↔ Epistemic: {corr_aleat_epist:.4f}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Compare Uncertainty Methods')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--output', type=str, default=None, help='Output folder')
    parser.add_argument('--mc_samples', type=int, default=3, help='Number of MC dropout samples (default: 3, reduce if OOM)')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Dropout probability for MC Dropout (default: 0.2)')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # GPU 메모리 완전 정리 및 최적화 설정
    if torch.cuda.is_available():
        # 강제 메모리 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        print(f"\n💾 GPU Memory Status:")
        print(f"  - Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  - Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  - Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  - Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB")
        print(f"\n⚙️  Memory Optimization:")
        print(f"  - MC Dropout samples: {args.mc_samples}")
        print(f"  - MC Dropout: All computations offloaded to CPU")
        print(f"  - Entropy calculation: Performed on CPU")
        print(f"  - Only final results stored on GPU")
        print(f"  - Tip: If OOM occurs, reduce --mc_samples (current: {args.mc_samples})")
        print(f"  - Or set: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print()

    # ===== 경로 설정 =====
    preprocessed_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/nnUNetPlans_3d_fullres'
    dataset_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/dataset.json'
    plans_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/nnUNetPlans.json'
    splits_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/splits_final.json'

    if args.output is None:
        output_folder = '/home/yoonji/AnatoMask/uncertainty_comparison'
    else:
        output_folder = args.output

    maybe_mkdir_p(output_folder)

    # ===== 데이터 로드 =====
    print("\n📚 Loading data...")

    dataset_json = load_json(dataset_json_path)
    plans = load_json(plans_json_path)
    splits = load_json(splits_path)

    all_keys = splits[args.fold]['train']
    from sklearn.model_selection import train_test_split
    tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

    dataset_val = nnUNetDataset(
        preprocessed_path, val_keys,
        folder_with_segs_from_previous_stage=None,
        num_images_properties_loading_threshold=0
    )
    

    print(f"✓ Validation samples: {len(dataset_val)} keys")

    # ===== 플랜 및 설정 =====
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    label_manager = plans_manager.get_label_manager(dataset_json)

    patch_size = configuration_manager.patch_size
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]

    rotation_for_DA = {
        'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    }

    initial_patch_size = get_patch_size(patch_size, *rotation_for_DA.values(), (0.85, 1.25))

    # Deep supervision scales 계산
    deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
        pool_op_kernel_sizes), axis=0))[:-1]

    print(f"✓ Patch size: {patch_size}")
    print(f"✓ Initial patch size: {initial_patch_size}")
    print(f"✓ Deep supervision scales: {deep_supervision_scales}")

    # ===== 모델 로드 =====
    print("\n🧠 Loading model...")

    # train_stunet_aleatoric.py와 동일한 모델 초기화
    # NOTE: Dropout layers are added for MC Dropout (p=0.2)
    model = SimpleUNetWithVarianceHead(in_channels=1, num_classes=105, base_channels=32)
    model = model.to(device)
    model.eval()

    # Count dropout layers
    dropout_count = sum(1 for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)))
    print(f"✓ Model created with {dropout_count} Dropout3d layers (p=0.2) for MC Dropout")

    if args.model_path is not None:
        print(f"Loading weights from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

        # train_stunet_aleatoric.py에서 저장한 checkpoint는 'network_weights' 키를 가짐
        if isinstance(checkpoint, dict) and 'network_weights' in checkpoint:
            print("Loading from checkpoint['network_weights']")
            model.load_state_dict(checkpoint['network_weights'], strict=False)
            print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print("Loading weights directly (no 'network_weights' key)")
            model.load_state_dict(checkpoint, strict=False)
        print("✓ Model weights loaded successfully (strict=False)")
        print("\n⚠️  NOTE: Dropout layers were added AFTER training for MC Dropout inference.")
        print("   These layers were not present during training, so epistemic uncertainty")
        print("   represents sampling uncertainty rather than model uncertainty.")
        print("   For better epistemic uncertainty, consider retraining with Dropout.")

    # ===== Data Loader 설정 (nnUNetDataLoader3D 사용) =====
    print("\n⚙️  Setting up validation data loader...")

    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

    # Validation Transform 정의 (deep_supervision_scales 적용)
    from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
    from batchgenerators.transforms.abstract_transforms import Compose

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    val_transforms.append(RenameTransform('seg', 'target', True))
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    # Data loader 생성
    dl_val = nnUNetDataLoader3D(
        dataset_val,
        batch_size=1,  # 한 번에 1개 샘플만
        patch_size=initial_patch_size,
        final_patch_size=patch_size,
        label_manager=label_manager,
        oversample_foreground_percent=0,  # Validation은 oversampling 없음
        sampling_probabilities=None,
        pad_sides=None
    )

    # ===== 비교 분석 =====
    num_samples_to_process = min(args.num_samples, len(val_keys))
    print(f"\n🔍 Comparing uncertainty methods ({num_samples_to_process} samples)...")
    print(f"✓ Data loader ready with batch size 1")

    try:
        allowed_num_processes = get_allowed_n_proc_DA()

        # LimitedLenWrapper 사용 (multithreaded augmentation)
        from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper

        mt_gen_val = LimitedLenWrapper(
            num_samples_to_process,
            data_loader=dl_val,
            transform=val_transforms,
            num_processes=allowed_num_processes,
            num_cached=2,
            seeds=None,
            pin_memory=True,
            wait_time=0.02
        )

        for sample_idx in range(num_samples_to_process):
            print(f"\n{'='*60}")
            print(f"Processing sample {sample_idx + 1}/{num_samples_to_process}")
            print(f"{'='*60}")

            # 샘플 시작 전 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"💾 GPU Memory before sample: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")

            try:
                # Data loader에서 배치 가져오기
                batch = next(mt_gen_val)

                inp = batch['data']  # [B, C, H, W, D] or list
                target = batch['target']  # [B, H, W, D] or list

                # Padding 함수 정의 (train_stunet_aleatoric.py와 동일)
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

                # Handle list case by padding
                if isinstance(inp, (list, tuple)):
                    print(f"  [INFO] inp is list, padding to common size")
                    inp = pad_tensors_to_common_size(inp)

                if isinstance(target, (list, tuple)):
                    print(f"  [INFO] target is list, using last element (full resolution)")
                    # For deep supervision, use last element only
                    target = target[-1] if len(target) > 0 else target[0]

                # Tensor로 변환 (이미 Tensor일 수 있음)
                if isinstance(inp, np.ndarray):
                    inp = torch.from_numpy(inp).float()
                elif not isinstance(inp, torch.Tensor):
                    inp = torch.tensor(inp).float()
                else:
                    inp = inp.float()

                if isinstance(target, np.ndarray):
                    target = torch.from_numpy(target).long()
                elif not isinstance(target, torch.Tensor):
                    target = torch.tensor(target).long()
                else:
                    target = target.long()

                # Device로 이동
                inp = inp.to(device)
                target = target.to(device)

                print(f"Input shape: {inp.shape}")
                print(f"Target shape: {target.shape}")

                # Forward pass (입력 데이터는 이미 patch_size로 설정됨)
                with torch.no_grad():
                    output = model(inp)

                    seg_logits = output['seg_logits']  # [B, C, H, W, D]
                    log_var = output['log_var']  # [B, 1, H, W, D]

                    # 1. Variance-based uncertainty
                    var_uncertainty = torch.exp(log_var)  # [B, 1, H, W, D]

                print(f"Running MC Dropout with {args.mc_samples} samples...")

                # 2. MC Dropout-based uncertainties
                # MC dropout을 한 번만 수행하고 두 uncertainty를 계산
                logit_list = mc_dropout(model, inp, T=args.mc_samples)  # T회 샘플링 (returns CPU tensors)

                # Aleatoric uncertainty (Entropy-based)
                aleatoric_uncertainty = calculate_aleatoric_uncertainty(logit_list, device=device)  # Returns GPU tensor

                # Epistemic uncertainty (Variance-based)
                epistemic_uncertainty = calculate_epistemic_uncertainty(logit_list, device=device)  # Returns GPU tensor

                # CPU 메모리 정리
                del logit_list
                torch.cuda.empty_cache()

                # Segmentation
                with torch.no_grad():
                    segmentation = seg_logits.argmax(dim=1)  # [B, H, W, D]

                print(f"\nOutput shapes:")
                print(f"  - seg_logits: {seg_logits.shape}")
                print(f"  - var_uncertainty (learned): {var_uncertainty.shape}")
                print(f"  - aleatoric_uncertainty (MC): {aleatoric_uncertainty.shape}")
                print(f"  - epistemic_uncertainty (MC): {epistemic_uncertainty.shape}")

                # 통계정보 출력
                print_statistics(var_uncertainty, aleatoric_uncertainty, epistemic_uncertainty,
                               label="Uncertainty Statistics:")

                # 시각화
                save_path = join(output_folder, f'comparison_{sample_idx:03d}.png')

                visualize_uncertainty_comparison(
                    inp,  # original image
                    seg_logits,
                    var_uncertainty,
                    aleatoric_uncertainty,
                    epistemic_uncertainty,
                    segmentation,
                    slice_idx=None,
                    save_path=save_path,
                    sample_name=f"sample_{sample_idx}"
                )

                # GPU 메모리 정리 (더 철저하게)
                del inp, target, output, seg_logits, log_var
                del var_uncertainty, aleatoric_uncertainty, epistemic_uncertainty, segmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            except Exception as e:
                print(f"✗ Error processing sample {sample_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                # 에러 발생 시에도 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                continue

        print(f"\n{'='*60}")
        print(f"✓ Comparison complete!")
        print(f"✓ Saved to: {output_folder}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL ERROR: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("\nCleaning up GPU memory...")
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("✓ GPU memory cleanup complete")


if __name__ == '__main__':
    main()
