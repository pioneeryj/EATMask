"""
Validation Set에서 Uncertainty Map을 시각화하는 스크립트

- nnUNetDataset에서 validation 데이터 로드 (augmentation 없음)
- 모델의 출력에서 uncertainty (exp(log_var)) 추출
- 중간 slice를 시각화
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
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, join
from nnunetv2.training.nnUNetTrainer.STUNetVarianceTrainer import VarianceHead, CombinedSegmentationVarianceLoss

# SimpleUNet 정의
class ConvBlock3D(nn.Module):
    """3D Conv Block: Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SimpleUNet3D(nn.Module):
    """간단한 3D U-Net"""
    def __init__(self, in_channels=1, num_classes=105, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock3D(in_channels, base_channels),
            ConvBlock3D(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            ConvBlock3D(base_channels, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool3d(2)

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

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock3D(base_channels * 8, base_channels * 4),
            ConvBlock3D(base_channels * 4, base_channels * 4)
        )

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock3D(base_channels * 4, base_channels * 2),
            ConvBlock3D(base_channels * 2, base_channels * 2)
        )

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock3D(base_channels * 2, base_channels),
            ConvBlock3D(base_channels, base_channels)
        )

        # Output
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.upconv3(x)

        # Skip connection 크기 맞추기
        if x.shape[2:] != enc3.shape[2:]:
            x = F.interpolate(x, size=enc3.shape[2:], mode='nearest')

        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)

        # Skip connection 크기 맞추기
        if x.shape[2:] != enc2.shape[2:]:
            x = F.interpolate(x, size=enc2.shape[2:], mode='nearest')

        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)

        # Skip connection 크기 맞추기
        if x.shape[2:] != enc1.shape[2:]:
            x = F.interpolate(x, size=enc1.shape[2:], mode='nearest')

        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        # Output
        x = self.final_conv(x)
        return x


class SimpleUNetWithVarianceHead(nn.Module):
    """SimpleUNet + Variance Head"""
    def __init__(self, base_model, num_classes=105):
        super().__init__()
        self.network = base_model
        self.variance_head = VarianceHead(num_classes=num_classes)

    def forward(self, x):
        seg_logits = self.network(x)
        log_var = self.variance_head(seg_logits)

        return {
            'seg_logits': seg_logits,
            'log_var': log_var
        }


def get_validation_transforms_simple(use_mask_for_norm=None):
    """
    검증용 transform - augmentation 없음, 기본 전처리만
    """
    from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

    val_transforms = []

    # Remove background label
    val_transforms.append(RemoveLabelTransform(-1, 0))

    # Rename seg to target (nnUNet 호환성)
    val_transforms.append(RenameTransform('seg', 'target', True))

    # Convert to Tensor
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    from batchgenerators.transforms.abstract_transforms import Compose
    return Compose(val_transforms)


def resize_data_for_model(data, target_size):
    """
    모델에 적합하도록 데이터 resize

    Args:
        data: [B, C, H, W, D] 형태의 데이터
        target_size: 목표 spatial size

    Returns:
        resize된 데이터
    """
    current_size = data.shape[2:]

    if current_size != target_size:
        data = F.interpolate(
            data,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

    return data


def visualize_uncertainty_slice(uncertainty_map, segmentation, slice_idx, class_names=None,
                                save_path=None, sample_name="sample"):
    """
    불확실성 맵의 중간 slice를 시각화

    Args:
        uncertainty_map: [1, 1, H, W, D] 또는 [H, W, D]
        segmentation: [1, H, W, D] 또는 [H, W, D]
        slice_idx: 시각화할 slice 인덱스
        class_names: 클래스 이름 (선택)
        save_path: 저장 경로
        sample_name: 샘플 이름
    """
    # 차원 정리
    if uncertainty_map.ndim == 5:
        uncertainty_map = uncertainty_map.squeeze(0).squeeze(0)  # [H, W, D]
    elif uncertainty_map.ndim == 4:
        uncertainty_map = uncertainty_map.squeeze(0)  # [H, W, D]

    if segmentation.ndim == 4:
        segmentation = segmentation.squeeze(0)  # [H, W, D]

    # CPU로 이동
    if isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.cpu().numpy()
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # 중간 slice 추출 (D 방향)
    mid_slice = uncertainty_map.shape[2] // 2 if slice_idx is None else slice_idx

    uncertainty_slice = uncertainty_map[:, :, mid_slice]  # [H, W]
    seg_slice = segmentation[:, :, mid_slice]  # [H, W]

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 불확실성 맵
    im1 = axes[0].imshow(uncertainty_slice, cmap='hot')
    axes[0].set_title(f'Uncertainty Map (Slice {mid_slice})\nexp(log_var)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Uncertainty')

    # 분할 맵
    im2 = axes[1].imshow(seg_slice, cmap='tab20')
    axes[1].set_title(f'Segmentation (Slice {mid_slice})', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Class')

    # 불확실성 + 분할 오버레이
    axes[2].imshow(seg_slice, cmap='gray', alpha=0.5)
    im3 = axes[2].imshow(uncertainty_slice, cmap='hot', alpha=0.5)
    axes[2].set_title(f'Uncertainty + Segmentation\n(Slice {mid_slice})', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], label='Uncertainty')

    plt.tight_layout()

    # 저장
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Visualize Uncertainty Maps')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--output', type=str, default=None, help='Output folder')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== 경로 설정 =====
    preprocessed_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/nnUNetPlans_3d_fullres'
    dataset_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/dataset.json'
    plans_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/nnUNetPlans.json'
    splits_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset606_all_TotalSegmentator/splits_final.json'

    if args.output is None:
        output_folder = '/home/yoonji/AnatoMask/uncertainty_visualizations'
    else:
        output_folder = args.output

    maybe_mkdir_p(output_folder)

    # ===== 데이터 로드 =====
    print("\n📚 Loading data...")

    dataset_json = load_json(dataset_json_path)
    plans = load_json(plans_json_path)
    splits = load_json(splits_path)

    val_keys = splits[args.fold]['val']

    dataset_val = nnUNetDataset(
        preprocessed_path, val_keys,
        folder_with_segs_from_previous_stage=None,
        num_images_properties_loading_threshold=0
    )

    print(f"✓ Validation samples: {len(dataset_val)}")

    # ===== 플랜 및 설정 =====
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    label_manager = plans_manager.get_label_manager(dataset_json)

    patch_size = configuration_manager.patch_size

    # ===== 모델 로드 =====
    print("\n🧠 Loading model...")

    unet = SimpleUNet3D(in_channels=1, num_classes=105, base_channels=32)
    model = SimpleUNetWithVarianceHead(unet, num_classes=105)
    model = model.to(device)
    model.eval()

    if args.model_path is not None:
        print(f"Loading weights from: {args.model_path}")
        weights = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(weights, strict=False)

    # ===== Data Loader 설정 (nnUNetDataLoader3D 사용) =====
    print("\n⚙️  Setting up validation data loader...")

    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
    from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
    from batchgenerators.transforms.abstract_transforms import Compose

    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
    rotation_for_DA = {
        'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    }

    initial_patch_size = get_patch_size(patch_size, *rotation_for_DA.values(), (0.85, 1.25))
    deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
        pool_op_kernel_sizes), axis=0))[:-1]

    # Validation Transform 정의
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    val_transforms.append(RenameTransform('seg', 'target', True))
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    # Data loader 생성
    dl_val = nnUNetDataLoader3D(
        dataset_val,
        batch_size=1,
        patch_size=initial_patch_size,
        final_patch_size=patch_size,
        label_manager=label_manager,
        oversample_foreground_percent=0,
        sampling_probabilities=None,
        pad_sides=None
    )

    # 각 샘플마다 처리
    num_samples_to_process = min(args.num_samples, len(val_keys))
    print(f"\n🔍 Visualizing uncertainty maps ({num_samples_to_process} samples)...")

    try:
        allowed_num_processes = get_allowed_n_proc_DA()

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
            print(f"\nProcessing sample {sample_idx + 1}/{num_samples_to_process}")

            try:
                # Data loader에서 배치 가져오기
                batch = next(mt_gen_val)

                inp = batch['data']  # [1, 1, H, W, D]
                target = batch['target']  # [1, H, W, D]

                # Tensor로 변환 (이미 Tensor일 수 있음)
                if isinstance(inp, np.ndarray):
                    inp = torch.from_numpy(inp).float()
                else:
                    inp = inp.float()

                if isinstance(target, np.ndarray):
                    target = torch.from_numpy(target).long()
                elif isinstance(target, (list, tuple)):
                    # Deep supervision: list인 경우 최고 해상도만 사용
                    target = target[-1]
                    if isinstance(target, np.ndarray):
                        target = torch.from_numpy(target).long()
                else:
                    target = target.long()

                # Device로 이동
                inp = inp.to(device)
                target = target.to(device)

                print(f"  - Input shape: {inp.shape}")

                # Forward pass (이미 patch_size로 설정됨)
                with torch.no_grad():
                    output = model(inp)

                    seg_logits = output['seg_logits']  # [B, C, H, W, D]
                    log_var = output['log_var']  # [B, 1, H, W, D]

                    # Uncertainty = exp(log_var)
                    uncertainty = torch.exp(log_var)  # [B, 1, H, W, D]

                    # Segmentation
                    segmentation = seg_logits.argmax(dim=1)  # [B, H, W, D]

                print(f"  - Output seg_logits shape: {seg_logits.shape}")
                print(f"  - Output log_var shape: {log_var.shape}")
                print(f"  - Uncertainty shape: {uncertainty.shape}")
                print(f"  - Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")

                # 시각화
                save_path = join(output_folder, f'uncertainty_map_{sample_idx:03d}.png')

                visualize_uncertainty_slice(
                    uncertainty,
                    segmentation,
                    slice_idx=None,
                    save_path=save_path,
                    sample_name=f"sample_{sample_idx}"
                )

                # GPU 메모리 정리
                del inp, target, output, seg_logits, log_var, uncertainty, segmentation
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ✗ Error processing sample {sample_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue

        print(f"\n{'='*60}")
        print(f"✓ Visualization complete!")
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
        # GPU 메모리 정리
        print("\nCleaning up GPU memory...")
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("✓ GPU memory cleanup complete")


if __name__ == '__main__':
    main()
