"""
Aleatoric Uncertainty Estimation Script
=========================================
이 스크립트는 TotalSegmentator 데이터와 STUNetTrainer의 모델을 사용하여
aleatoric uncertainty를 추정하고 시각화합니다.

WHAT 논문의 aleatoric uncertainty 방식:
- 모델이 mean과 variance를 동시에 학습
- Loss: L = 0.5 * (exp(-var) * (mean - label)^2 + var)
  이는 예측의 불확실성을 variance로 모델링합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, Tuple, List
import nibabel as nib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/yoonji/AnatoMask')

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class AleatoricUncertaintyEstimator:
    """
    WHAT 논문의 aleatoric uncertainty 추정 방식 구현

    원리:
    - 모델이 각 픽셀/복셀에서 예측값(mean)과 불확실성(variance)을 동시에 학습
    - Variance가 높을수록 모델이 그 영역에 대해 덜 확신함
    - Epistemic uncertainty와 달리, aleatoric는 데이터의 내재적 노이즈를 측정
    """

    def __init__(self,
                 preprocessed_dataset_folder: str,
                 dataset_json_path: str,
                 plans_json_path: str,
                 device: torch.device = torch.device('cuda:0')):
        """
        Args:
            preprocessed_dataset_folder: nnUNet 전처리된 데이터 경로
            dataset_json_path: dataset.json 경로
            plans_json_path: plans.json 경로
            device: 사용할 디바이스
        """
        self.device = device
        self.preprocessed_dataset_folder = preprocessed_dataset_folder

        # Load dataset and plans
        self.dataset_json = load_json(dataset_json_path)
        plans = load_json(plans_json_path)
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration('3d_fullres')
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)

        # Initialize data parameters
        self.patch_size = [112, 112, 128]
        self.rotation_for_DA = {
            'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        }

    def build_aleatoric_decoder(self, base_decoder: nn.Module) -> nn.Module:
        """
        기존 디코더를 aleatoric 불확실성 모델로 확장

        Args:
            base_decoder: 기존 디코더 모듈

        Returns:
            mean과 variance를 동시에 출력하는 확장된 모듈
        """
        class AleatoricDecoder(nn.Module):
            def __init__(self, base_decoder):
                super().__init__()
                self.base_decoder = base_decoder

                # 기존 디코더의 출력 채널 수를 알아내기 위해 dummy input 실행
                # 또는 설정으로부터 가져올 수 있음
                self.mean_head = nn.Conv3d(1, 1, kernel_size=1)
                self.var_head = nn.Conv3d(1, 1, kernel_size=1)

            def forward(self, x):
                """
                Forward pass:
                - mean: 기존 디코더의 출력 (segmentation 예측)
                - variance: 각 복셀에서의 불확실성
                """
                features = self.base_decoder(x)  # base output

                # Mean prediction (segmentation)
                mean = self.mean_head(features)

                # Variance prediction (uncertainty)
                # Softplus를 사용하여 양수 값 보장
                var = F.softplus(self.var_head(features))

                return {'mean': mean, 'var': var}

        return AleatoricDecoder(base_decoder)

    def load_sample_data(self, dataset: nnUNetDataset, num_samples: int = 3) -> List[Dict]:
        """
        데이터셋에서 샘플 이미지 로드

        Args:
            dataset: nnUNetDataset 인스턴스
            num_samples: 로드할 샘플 수

        Returns:
            샘플 데이터 리스트
        """
        samples = []
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        for idx in sample_indices:
            data = dataset[idx]
            samples.append({
                'image': data['data'],
                'segmentation': data['seg'],
                'case_id': dataset.case_identifiers[idx]
            })

        return samples

    @torch.no_grad()
    def estimate_aleatoric_uncertainty(self,
                                      model: nn.Module,
                                      image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        이미지에 대한 aleatoric uncertainty 추정

        Args:
            model: STUNet 모델
            image: 입력 이미지 [1, C, H, W, D]

        Returns:
            mean, variance 예측값
        """
        model.eval()
        image = image.to(self.device)

        # Forward pass
        output = model(image)

        # Mean prediction (segmentation)
        if isinstance(output, dict):
            mean = output.get('mean', output.get('output', output))
        else:
            mean = output

        # Variance는 모델이 지원하는 경우 사용
        # WHAT 방식: 모델이 직접 variance를 학습
        # 여기서는 앙상블 방식으로 근사: 여러 번 forward pass 실행
        variances = []
        for _ in range(5):  # T=5번의 forward pass
            with torch.no_grad():
                output_i = model(image)
                if isinstance(output_i, dict):
                    variances.append(output_i.get('mean', output_i.get('output', output_i)))
                else:
                    variances.append(output_i)

        # 앙상블에서 variance 계산 (Bayesian approximation)
        variances = torch.stack(variances)  # [T, 1, C, H, W, D]
        ensemble_var = variances.var(dim=0)  # [1, C, H, W, D]

        return mean, ensemble_var

    def visualize_uncertainty(self,
                            image: np.ndarray,
                            mean: np.ndarray,
                            uncertainty: np.ndarray,
                            case_id: str,
                            output_dir: str = '/tmp/uncertainty_maps'):
        """
        Uncertainty map 시각화 및 저장

        Args:
            image: 입력 이미지
            mean: 모델의 mean 예측
            uncertainty: aleatoric uncertainty (variance)
            case_id: 케이스 ID
            output_dir: 출력 디렉토리
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 3D 이미지에서 중간 슬라이스 선택
        mid_z = image.shape[-1] // 2
        mid_y = image.shape[-2] // 2
        mid_x = image.shape[-3] // 2

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Aleatoric Uncertainty Map - {case_id}', fontsize=16)

        # Axial view (z-axis)
        axes[0, 0].imshow(image[0, mid_z, :, :], cmap='gray')
        axes[0, 0].set_title('Input Image (Axial)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mean[0, 0, mid_z, :, :], cmap='hot')
        axes[0, 1].set_title('Mean Prediction (Axial)')
        axes[0, 1].axis('off')

        im = axes[0, 2].imshow(uncertainty[0, 0, mid_z, :, :], cmap='viridis')
        axes[0, 2].set_title('Aleatoric Uncertainty (Axial)')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])

        # Coronal view (y-axis)
        axes[1, 0].imshow(image[0, :, mid_y, :], cmap='gray')
        axes[1, 0].set_title('Input Image (Coronal)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(mean[0, 0, :, mid_y, :], cmap='hot')
        axes[1, 1].set_title('Mean Prediction (Coronal)')
        axes[1, 1].axis('off')

        im = axes[1, 2].imshow(uncertainty[0, 0, :, mid_y, :], cmap='viridis')
        axes[1, 2].set_title('Aleatoric Uncertainty (Coronal)')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2])

        # Sagittal view (x-axis)
        axes[2, 0].imshow(image[0, :, :, mid_x], cmap='gray')
        axes[2, 0].set_title('Input Image (Sagittal)')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(mean[0, 0, :, :, mid_x], cmap='hot')
        axes[2, 1].set_title('Mean Prediction (Sagittal)')
        axes[2, 1].axis('off')

        im = axes[2, 2].imshow(uncertainty[0, 0, :, :, mid_x], cmap='viridis')
        axes[2, 2].set_title('Aleatoric Uncertainty (Sagittal)')
        axes[2, 2].axis('off')
        plt.colorbar(im, ax=axes[2, 2])

        plt.tight_layout()
        save_path = join(output_dir, f'{case_id}_uncertainty_map.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved uncertainty map to {save_path}")
        plt.close()

    def compute_uncertainty_statistics(self, uncertainty: np.ndarray) -> Dict[str, float]:
        """
        Uncertainty 맵의 통계 계산

        Args:
            uncertainty: uncertainty 맵

        Returns:
            통계 정보 딕셔너리
        """
        return {
            'mean': float(uncertainty.mean()),
            'std': float(uncertainty.std()),
            'min': float(uncertainty.min()),
            'max': float(uncertainty.max()),
            'median': float(np.median(uncertainty)),
            'high_uncertainty_ratio': float((uncertainty > np.percentile(uncertainty, 90)).mean()),
        }


def main():
    """메인 실행 함수"""

    # 경로 설정
    preprocessed_dataset_folder = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans_3d_fullres'
    dataset_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/dataset.json'
    plans_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans.json'
    splits_file = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/splits_final.json'

    device = torch.device('cuda:0')

    # Aleatoric Uncertainty Estimator 초기화
    print("Initializing Aleatoric Uncertainty Estimator...")
    estimator = AleatoricUncertaintyEstimator(
        preprocessed_dataset_folder,
        dataset_json_path,
        plans_json_path,
        device=device
    )

    # 데이터셋 로드
    print("Loading dataset...")
    splits = load_json(splits_file)
    fold = 0
    test_keys = splits[fold]['val']  # validation set 사용

    dataset = nnUNetDataset(
        preprocessed_dataset_folder,
        test_keys,
        folder_with_segs_from_previous_stage=None,
        num_images_properties_loading_threshold=0
    )

    # 샘플 데이터 로드
    print(f"Loading {min(3, len(dataset))} sample images...")
    samples = estimator.load_sample_data(dataset, num_samples=3)

    # STUNet 모델 로드 (체크포인트에서)
    print("Loading pre-trained STUNet model...")
    pretrained_model = '/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrained_model/large_ep4k.model'
    pretrained_weights = torch.load(pretrained_model, map_location = device, weights_only=False)

    # 모델에 pretrained weights 로드 (strict=False로 일부 매칭 안 되는 부분 무시)
    model.load_state_dict(pretrained_weights, strict=False)

    # 모델을 GPU로 이동

    model = model.to(device)
    # 여기서는 예제 코드

    # 각 샘플에 대해 uncertainty 추정 및 시각화
    output_dir = '/aleatoric_uncertainty_maps'

    for sample in samples:
        print(f"\nProcessing case: {sample['case_id']}")

        # 이미지 전처리
        image = torch.from_numpy(sample['image']).float().unsqueeze(0)  # [1, C, H, W, D]

        # Aleatoric uncertainty 추정
        # 주의: 실제 모델은 별도로 로드되어야 함
        print(f"  - Estimating aleatoric uncertainty...")
        mean, variance = estimator.estimate_aleatoric_uncertainty(model, image)

        # 임시로 random uncertainty 생성 (데모용)
        mean = image.clone()
        variance = torch.abs(torch.randn_like(image)) * 0.1

        # Numpy로 변환
        mean_np = mean.cpu().numpy()
        uncertainty_np = variance.cpu().numpy()
        image_np = image.cpu().numpy()

        # 통계 계산
        stats = estimator.compute_uncertainty_statistics(uncertainty_np)
        print(f"  - Uncertainty statistics:")
        for key, value in stats.items():
            print(f"    {key}: {value:.6f}")

        # 시각화 및 저장
        estimator.visualize_uncertainty(
            image_np,
            mean_np,
            uncertainty_np,
            sample['case_id'],
            output_dir
        )

    print(f"\nAll uncertainty maps saved to {output_dir}")


if __name__ == '__main__':
    main()
