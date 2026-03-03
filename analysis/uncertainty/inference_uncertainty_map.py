"""
STUNet Aleatoric Uncertainty Map 추출
======================================

train_stunet_aleatoric.py에서 학습된 모델을 로드해서
WHAT 논문 방식의 variance 기반 aleatoric uncertainty map을 생성합니다.

사용법:
    # 기본 실행
    python inference_uncertainty_map.py --checkpoint /path/to/stunet_aleatoric_best.pt --fold 0

    # 커스텀 출력
    python inference_uncertainty_map.py \
        --checkpoint /path/to/stunet_aleatoric_best.pt \
        --fold 0 \
        --output /custom/output/path \
        --num_samples 5
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import argparse
from tqdm import tqdm

sys.path.insert(0, '/home/yoonji/AnatoMask')

from STUNet_head import STUNet
from train_stunet_aleatoric import STUNetWithUncertainty
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from batchgenerators.utilities.file_and_folder_operations import load_json, join, maybe_mkdir_p


class UncertaintyMapGenerator:
    """
    학습된 STUNet 모델에서 Aleatoric Uncertainty Map을 생성합니다.

    WHAT 논문 방식:
    - 모델이 각 복셀에서 mean (분할)과 variance (불확실성)을 동시에 예측
    - Aleatoric uncertainty = variance (데이터의 내재적 노이즈)
    - 불확실성이 높을수록 모델이 그 영역의 예측에 확신이 없음
    """

    def __init__(self,
                 checkpoint_path: str,
                 fold: int = 0,
                 preprocessed_data_path: str = None,
                 dataset_json_path: str = None,
                 device: str = 'cuda:0'):
        """
        Args:
            checkpoint_path: 학습된 모델 checkpoint 경로
            fold: 폴드 번호
            preprocessed_data_path: 전처리된 데이터 경로
            dataset_json_path: dataset.json 경로
            device: 사용할 디바이스
        """
        self.device = torch.device(device)
        self.fold = fold
        self.checkpoint_path = checkpoint_path

        # 기본 경로 설정
        if preprocessed_data_path is None:
            preprocessed_data_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans_3d_fullres'
        if dataset_json_path is None:
            dataset_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/dataset.json'

        self.preprocessed_data_path = preprocessed_data_path
        self.dataset_json_path = dataset_json_path

        # 데이터셋 로드
        print(f"📂 Loading dataset...")
        self.dataset_json = load_json(dataset_json_path)

        # Splits 로드
        splits_path = join(Path(preprocessed_data_path).parent, 'splits_final.json')
        if os.path.exists(splits_path):
            self.splits = load_json(splits_path)
            self.test_keys = self.splits[fold]['val']
        else:
            print(f"⚠️  splits_final.json not found at {splits_path}")
            self.test_keys = None

        # 모델 구축 및 로드
        print(f"🧠 Building STUNet model...")
        self.model = self._build_and_load_model()

        print(f"✓ Model loaded from {checkpoint_path}")

    def _build_and_load_model(self) -> nn.Module:
        """
        STUNet 모델 구축 및 checkpoint 로드
        train_stunet_aleatoric.py와 동일한 구조
        """
        # STUNet 구축 (train_stunet_aleatoric.py의 설정과 동일)
        pool_op_kernel_sizes = [
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [1, 2, 2]
        ]
        conv_kernel_sizes = [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3]
        ]

        head = STUNet(
            1, 1,
            depth=[1, 1, 1, 1, 1, 1],
            dims=[32, 64, 128, 256, 512, 512],
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            conv_kernel_sizes=conv_kernel_sizes,
            enable_deep_supervision=True
        ).to(self.device)

        # Uncertainty 래퍼 추가
        model = STUNetWithUncertainty(head).to(self.device)

        # Checkpoint 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # state_dict 호환성 처리 (DDP에서 'module.' prefix 제거)
        if 'network_weights' in checkpoint:
            state_dict = checkpoint['network_weights']
        else:
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # 'module.' prefix 제거 (DDP에서 저장된 경우)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 'module.' 제거
            else:
                new_key = key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        return model

    @torch.no_grad()
    def extract_uncertainty_map(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        입력 이미지에서 uncertainty map 추출

        Args:
            image: [1, C, H, W, D] 입력 이미지

        Returns:
            mean: 분할 예측 [1, 1, H, W, D]
            var: aleatoric uncertainty map [1, 1, H, W, D]
        """
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)

        mean = output['mean']  # 분할
        var = output['var']    # aleatoric uncertainty

        return mean, var

    def compute_uncertainty_statistics(self,
                                      uncertainty: np.ndarray,
                                      mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Uncertainty 맵의 통계 계산

        Args:
            uncertainty: Uncertainty map [1, H, W, D] or [H, W, D]
            mask: 관심 영역 마스크 (선택)

        Returns:
            통계 정보
        """
        # 차원 정리
        if uncertainty.ndim == 4:
            uncertainty = uncertainty[0]

        # 마스크 적용
        if mask is not None:
            if mask.ndim == 4:
                mask = mask[0]
            uncertainty_masked = uncertainty[mask > 0]
        else:
            uncertainty_masked = uncertainty

        stats = {
            'mean': float(np.mean(uncertainty_masked)),
            'std': float(np.std(uncertainty_masked)),
            'min': float(np.min(uncertainty_masked)),
            'max': float(np.max(uncertainty_masked)),
            'median': float(np.median(uncertainty_masked)),
            'q25': float(np.percentile(uncertainty_masked, 25)),
            'q75': float(np.percentile(uncertainty_masked, 75)),
            'q90': float(np.percentile(uncertainty_masked, 90)),
            'high_uncertainty_ratio': float((uncertainty_masked > np.percentile(uncertainty_masked, 90)).mean()),
        }
        return stats

    def visualize_uncertainty_3d(self,
                                image: np.ndarray,
                                segmentation: np.ndarray,
                                uncertainty: np.ndarray,
                                case_id: str,
                                output_dir: str):
        """
        3D 이미지의 Uncertainty Map 시각화

        3개의 뷰 (Axial, Coronal, Sagittal)에서 시각화

        Args:
            image: 입력 이미지 [C, H, W, D] or [H, W, D]
            segmentation: Segmentation 결과 [1, H, W, D] or [H, W, D]
            uncertainty: Uncertainty map [1, H, W, D] or [H, W, D]
            case_id: 케이스 ID
            output_dir: 출력 디렉토리
        """
        maybe_mkdir_p(output_dir)

        # 차원 정리
        if image.ndim == 4:
            image = image[0]
        if segmentation.ndim == 4:
            segmentation = segmentation[0]
        if uncertainty.ndim == 4:
            uncertainty = uncertainty[0]

        # 중간 슬라이스 선택
        z_mid = image.shape[-1] // 2
        y_mid = image.shape[0] // 2
        x_mid = image.shape[1] // 2

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'STUNet Aleatoric Uncertainty Map - {case_id}',
                    fontsize=16, fontweight='bold')

        # ========== Axial View (Z-axis) ==========
        ax1 = plt.subplot(3, 4, 1)
        im1 = ax1.imshow(image[y_mid, :, z_mid], cmap='gray')
        ax1.set_title('Input (Axial)', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = plt.subplot(3, 4, 2)
        im2 = ax2.imshow(segmentation[y_mid, :, z_mid], cmap='jet')
        ax2.set_title('Segmentation (Axial)', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = plt.subplot(3, 4, 3)
        im3 = ax3.imshow(uncertainty[y_mid, :, z_mid], cmap='viridis')
        ax3.set_title('Uncertainty (Axial)', fontweight='bold')
        ax3.axis('off')
        cbar = plt.colorbar(im3, ax=ax3, fraction=0.046)
        cbar.set_label('Aleatoric\nUncertainty')

        ax4 = plt.subplot(3, 4, 4)
        # Overlay: uncertainty on image
        masked_uncertainty = uncertainty[y_mid, :, z_mid] * (segmentation[y_mid, :, z_mid] > 0)
        im4 = ax4.imshow(image[y_mid, :, z_mid], cmap='gray')
        im4_overlay = ax4.imshow(masked_uncertainty, cmap='hot', alpha=0.6)
        ax4.set_title('Uncertainty Overlay (Axial)', fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4_overlay, ax=ax4, fraction=0.046)

        # ========== Coronal View (Y-axis) ==========
        ax5 = plt.subplot(3, 4, 5)
        im5 = ax5.imshow(image[:, x_mid, z_mid], cmap='gray')
        ax5.set_title('Input (Coronal)', fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)

        ax6 = plt.subplot(3, 4, 6)
        im6 = ax6.imshow(segmentation[:, x_mid, z_mid], cmap='jet')
        ax6.set_title('Segmentation (Coronal)', fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)

        ax7 = plt.subplot(3, 4, 7)
        im7 = ax7.imshow(uncertainty[:, x_mid, z_mid], cmap='viridis')
        ax7.set_title('Uncertainty (Coronal)', fontweight='bold')
        ax7.axis('off')
        cbar = plt.colorbar(im7, ax=ax7, fraction=0.046)
        cbar.set_label('Aleatoric\nUncertainty')

        ax8 = plt.subplot(3, 4, 8)
        masked_uncertainty = uncertainty[:, x_mid, z_mid] * (segmentation[:, x_mid, z_mid] > 0)
        im8 = ax8.imshow(image[:, x_mid, z_mid], cmap='gray')
        im8_overlay = ax8.imshow(masked_uncertainty, cmap='hot', alpha=0.6)
        ax8.set_title('Uncertainty Overlay (Coronal)', fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8_overlay, ax=ax8, fraction=0.046)

        # ========== Sagittal View (X-axis) ==========
        ax9 = plt.subplot(3, 4, 9)
        im9 = ax9.imshow(image[:, :, x_mid], cmap='gray')
        ax9.set_title('Input (Sagittal)', fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046)

        ax10 = plt.subplot(3, 4, 10)
        im10 = ax10.imshow(segmentation[:, :, x_mid], cmap='jet')
        ax10.set_title('Segmentation (Sagittal)', fontweight='bold')
        ax10.axis('off')
        plt.colorbar(im10, ax=ax10, fraction=0.046)

        ax11 = plt.subplot(3, 4, 11)
        im11 = ax11.imshow(uncertainty[:, :, x_mid], cmap='viridis')
        ax11.set_title('Uncertainty (Sagittal)', fontweight='bold')
        ax11.axis('off')
        cbar = plt.colorbar(im11, ax=ax11, fraction=0.046)
        cbar.set_label('Aleatoric\nUncertainty')

        ax12 = plt.subplot(3, 4, 12)
        masked_uncertainty = uncertainty[:, :, x_mid] * (segmentation[:, :, x_mid] > 0)
        im12 = ax12.imshow(image[:, :, x_mid], cmap='gray')
        im12_overlay = ax12.imshow(masked_uncertainty, cmap='hot', alpha=0.6)
        ax12.set_title('Uncertainty Overlay (Sagittal)', fontweight='bold')
        ax12.axis('off')
        plt.colorbar(im12_overlay, ax=ax12, fraction=0.046)

        plt.tight_layout()
        save_path = join(output_dir, f'{case_id}_uncertainty_map.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
        plt.close()

    def print_statistics(self, stats: Dict[str, float], case_id: str):
        """통계 정보 출력"""
        print(f"\n{'='*70}")
        print(f"Aleatoric Uncertainty Statistics - {case_id}")
        print(f"{'='*70}")
        print(f"  Mean:                    {stats['mean']:.6f}")
        print(f"  Std:                     {stats['std']:.6f}")
        print(f"  Min:                     {stats['min']:.6f}")
        print(f"  Max:                     {stats['max']:.6f}")
        print(f"  Median:                  {stats['median']:.6f}")
        print(f"  Q25-Q75:                 {stats['q25']:.6f} - {stats['q75']:.6f}")
        print(f"  Q90:                     {stats['q90']:.6f}")
        print(f"  High Uncertainty Ratio:  {stats['high_uncertainty_ratio']:.4f} ({stats['high_uncertainty_ratio']*100:.2f}%)")
        print(f"{'='*70}")

    def process_samples(self, num_samples: int = 5, output_dir: str = None):
        """
        테스트 샘플들을 처리해서 uncertainty map 생성

        Args:
            num_samples: 처리할 샘플 수
            output_dir: 출력 디렉토리
        """
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'/tmp/uncertainty_maps_{timestamp}'

        maybe_mkdir_p(output_dir)
        print(f"\n📁 Output directory: {output_dir}\n")

        # 데이터셋 로드
        if self.test_keys is None:
            print("⚠️  No test keys available, using first samples from dataset")
            dataset = nnUNetDataset(
                self.preprocessed_data_path,
                list(range(min(num_samples, 100))),
                folder_with_segs_from_previous_stage=None,
                num_images_properties_loading_threshold=0
            )
        else:
            test_keys = self.test_keys[:num_samples]
            dataset = nnUNetDataset(
                self.preprocessed_data_path,
                test_keys,
                folder_with_segs_from_previous_stage=None,
                num_images_properties_loading_threshold=0
            )

        print(f"🔄 Processing {min(num_samples, len(dataset))} samples...")

        for idx in range(min(num_samples, len(dataset))):
            print(f"\n[{idx+1}/{min(num_samples, len(dataset))}] Processing sample {idx}...")

            # 데이터 로드
            data = dataset[idx]
            image = torch.from_numpy(data['data']).float().unsqueeze(0)  # [1, C, H, W, D]
            segmentation = data['seg']

            # Uncertainty map 추출 (WHAT 방식: variance head)
            print(f"  - Extracting uncertainty map...")
            mean, var = self.extract_uncertainty_map(image)

            # CPU로 이동
            mean_np = mean.cpu().numpy()
            var_np = var.cpu().numpy()
            image_np = image.cpu().numpy()

            # 통계 계산
            stats = self.compute_uncertainty_statistics(var_np, segmentation)
            self.print_statistics(stats, f"Sample_{idx}")

            # 시각화
            print(f"  - Generating visualization...")
            self.visualize_uncertainty_3d(
                image_np,
                mean_np,
                var_np,
                f"Sample_{idx}",
                output_dir
            )

            # Uncertainty map 저장 (numpy 형식)
            uncertainty_save_path = join(output_dir, f'Sample_{idx}_uncertainty_map.npy')
            np.save(uncertainty_save_path, var_np)
            print(f"  ✓ Saved uncertainty map to {uncertainty_save_path}")

        print(f"\n✅ All samples processed! Results saved to {output_dir}")
        return output_dir


def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(
        description='STUNet Aleatoric Uncertainty Map 추출',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 실행
  python inference_uncertainty_map.py --checkpoint /path/to/stunet_aleatoric_best.pt

  # 5개 샘플 처리
  python inference_uncertainty_map.py --checkpoint /path/to/stunet_aleatoric_best.pt --num_samples 5

  # 커스텀 출력 폴더
  python inference_uncertainty_map.py --checkpoint /path/to/stunet_aleatoric_best.pt --output /custom/path
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/yoonji/AnatoMask/Anatomask_results/aleatoric_uncertainty/stunet_aleatoric/fold0/stunet_aleatoric_best.pt',
        help='학습된 모델 checkpoint 경로'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='폴드 번호 (0-4)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='처리할 샘플 수'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 디렉토리 (미지정시 자동 생성)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='사용할 디바이스'
    )

    args = parser.parse_args()

    # Checkpoint 확인
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"   Expected path: {args.checkpoint}")
        return

    print(f"\n{'='*70}")
    print(f"STUNet Aleatoric Uncertainty Map Extraction")
    print(f"{'='*70}")
    print(f"📦 Checkpoint: {args.checkpoint}")
    print(f"📊 Fold: {args.fold}")
    print(f"🔢 Num samples: {args.num_samples}")
    print(f"{'='*70}\n")

    # Generator 초기화
    generator = UncertaintyMapGenerator(
        checkpoint_path=args.checkpoint,
        fold=args.fold,
        device=args.device
    )

    # 샘플 처리
    generator.process_samples(
        num_samples=args.num_samples,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
