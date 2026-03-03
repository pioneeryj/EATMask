# Aleatoric Uncertainty Estimation Guide

## 개요

이 가이드는 WHAT (Uncertainty-aware Networks with Heteroscedastic Aleatory) 논문의 방식을 사용하여 TotalSegmentator 데이터셋에서 **Aleatoric Uncertainty**를 추정하고 시각화하는 방법을 설명합니다.

### Aleatoric vs Epistemic Uncertainty

| 유형 | 설명 | 원인 | 처리 방법 |
|------|------|------|---------|
| **Aleatoric** | 데이터의 내재적 노이즈 | 이미지 품질 저하, 애매한 경계 | 더 많은 데이터로 개선 불가 |
| **Epistemic** | 모델의 지식 부족 | 불충분한 학습 데이터 | MC Dropout, 앙상블로 감소 |

## WHAT 논문의 방식

### 핵심 원리

WHAT 논문은 모델이 각 픽셀/복셀에서 **두 가지를 동시에 학습**하는 방식입니다:

```
출력 = {
    'mean': 분할 예측값 (평균)
    'variance': 불확실성 (분산)
}
```

### 손실 함수

```
Loss = 0.5 * [exp(-variance) * (mean - target)^2 + variance]
     = 0.5 * [1/variance * MSE + variance]
```

**해석:**
- 높은 `variance` → 모델이 그 영역에 불확실함
- 낮은 `variance` → 모델이 자신감 있음
- 모델이 자동으로 각 영역의 불확실성을 학습

## 구현된 코드

### 1. `estimate_aleatoric_uncertainty.py`

기본적인 구조를 제공하는 버전입니다.

**주요 기능:**
- Aleatoric uncertainty 추정 기본 클래스
- MC Dropout을 사용한 불확실성 근사
- 3D 이미지 시각화

**사용:**
```python
python estimate_aleatoric_uncertainty.py
```

### 2. `evaluate_uncertainty_totalsegmentator.py` (권장)

STUNetTrainer와 TotalSegmentator 데이터셋과 통합된 완전한 버전입니다.

**주요 기능:**
- STUNetTrainer 모델과 직접 통합
- TotalSegmentator 데이터 로드
- 3개 뷰 (Axial, Coronal, Sagittal) 시각화
- 상세한 통계 계산
- 불확실성 맵 오버레이

## 사용 방법

### 기본 사용

```python
from evaluate_uncertainty_totalsegmentator import TotalSegmentatorUncertaintyEvaluator
from batchgenerators.utilities.file_and_folder_operations import load_json

# 경로 설정
preprocessed_data_path = '/path/to/preprocessed/data'
dataset_json_path = '/path/to/dataset.json'
plans_json_path = '/path/to/nnUNetPlans.json'

# 데이터 로드
dataset_json = load_json(dataset_json_path)
plans = load_json(plans_json_path)

# Evaluator 초기화
evaluator = TotalSegmentatorUncertaintyEvaluator(
    plans,
    '3d_fullres',
    dataset_json,
    device=torch.device('cuda:0')
)

# 네트워크 구축
network = evaluator.build_network()
network.eval()

# 이미지 로드
image = torch.randn(1, 1, 112, 112, 128).cuda()

# Uncertainty 추정
mean, variance = evaluator.estimate_uncertainty_mc_dropout(
    network,
    image,
    mc_iterations=10
)

# 시각화
evaluator.visualize_3d_uncertainty(
    image.cpu().numpy(),
    mean.cpu().numpy(),
    variance.cpu().numpy(),
    'case_001',
    output_dir='/tmp/results'
)
```

### MC Dropout vs Variance Head

#### MC Dropout 방식 (권장: 기존 모델 사용 가능)

```python
mean, variance = evaluator.estimate_uncertainty_mc_dropout(
    network,
    image,
    mc_iterations=10
)
```

**장점:**
- 기존 모델 재사용 가능
- 추가 학습 불필요
- 이론적 근거 있음 (Bayesian approximation)

**단점:**
- 여러 번의 forward pass 필요 (느림)
- 메모리 사용량 많음

**원리:**
```
Test time에 Dropout 활성화
→ T번의 forward pass 실행
→ 출력의 분산 = aleatoric uncertainty 근사
```

#### Variance Head 방식 (빠름, 새 학습 필요)

```python
network = evaluator.extend_with_uncertainty_head(network)
network = evaluator.build_network()  # Variance head 추가
# 학습 시 variance head도 함께 학습
mean, variance = evaluator.estimate_uncertainty_variance_head(network, image)
```

**장점:**
- 한 번의 forward pass (빠름)
- 직접 예측

**단점:**
- 모델 재학습 필요
- WHAT 손실함수로 학습해야 함

## 출력 해석

### 불확실성 맵

```
Uncertainty Map:
- 높은 값 (밝은 색): 높은 불확실성
- 낮은 값 (어두운 색): 낮은 불확실성
```

### 예상 패턴

**높은 불확실성이 나타나는 영역:**
1. 장기 경계 부근 (경계가 애매함)
2. 작은 장기 (학습 데이터 부족)
3. 질병이 있는 영역 (예측 어려움)
4. 이미지 아티팩트가 있는 영역

**낮은 불확실성이 나타나는 영역:**
1. 배경 (명확한 구분)
2. 큰 장기 내부 (일관된 신호)
3. 정상 해부학적 구조

### 통계 정보

```
Uncertainty Statistics:
  Mean:                  0.125000
  Std:                   0.045000
  Min:                   0.002000
  Max:                   0.523000
  Median:                0.110000
  Q25-Q75:               0.085000 - 0.155000
  High Uncertainty Ratio:0.1000 (10.00%)
```

**해석:**
- `Mean/Std`: 전체 불확실성 정도
- `High Uncertainty Ratio`: 상위 10% 임계값 이상인 복셀 비율
- 높을수록: 모델이 더 불확실함

## 시각화 이해

### 3개 뷰 구성

각 행은 3가지 뷰를 보여줍니다:

1. **Axial (위에서 본 뷰)**: Z축 방향
2. **Coronal (앞에서 본 뷰)**: Y축 방향
3. **Sagittal (옆에서 본 뷰)**: X축 방향

각 뷰의 4개 열:
- **Input**: 원본 이미지 (회색)
- **Segmentation**: 분할 결과 (색상)
- **Uncertainty**: 불확실성 맵 (파란색-노란색)
- **Overlay**: 분할 영역 위에 불확실성 표시

## 성능 최적화

### 메모리 사용 감소

```python
# 배치 크기 줄임
batch_size = 1  # 기본값

# MC iterations 줄임
mc_iterations = 5  # 기본값: 10
```

### 속도 향상

```python
# 1. Variance head 사용 (MC Dropout 대신)
# 2. 낮은 해상도에서 처리
# 3. GPU 여러 개 사용 (DDP)
```

## 실험 결과 해석

### 모델 평가

**높은 불확실성이 정상:**
- 경계 영역이 많은 경우
- 해부학적 변이가 큰 데이터셋

**낮은 불확실성이 정상:**
- 명확한 경계
- 정상 해부학

**이상 신호:**
- 균일하게 높은 불확실성: 모델이 제대로 학습되지 않음
- 예상치 못한 영역의 높은 불확실성: 이상 탐지

## 트러블슈팅

### 문제: 모든 영역에서 불확실성이 균일함

**원인:** 모델이 제대로 학습되지 않음

**해결책:**
```python
# 1. 모델 체크포인트 확인
# 2. 입력 정규화 확인
# 3. MC iterations 증가
mc_iterations = 20
```

### 문제: 불확실성이 너무 낮음

**원인:** 모델이 과적합됨

**해결책:**
```python
# 1. Dropout rate 확인
# 2. 정규화 강도 증가
# 3. 더 다양한 데이터 사용
```

### 문제: 메모리 부족 에러

**원인:** MC iterations이 너무 많음

**해결책:**
```python
# 1. MC iterations 줄임
mc_iterations = 5

# 2. 배치 크기 줄임
batch_size = 1

# 3. 입력 해상도 줄임
input_size = [96, 96, 96]  # 112x112x128 대신
```

## 참고 자료

### WHAT 논문
- "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
- 링크: https://arxiv.org/abs/1506.02640

### 관련 코드
- `/home/yoonji/AnatoMask/what/WHAT_src/`: WHAT 원본 구현
- `/home/yoonji/AnatoMask/estimate_aleatoric_uncertainty.py`: 기본 구현
- `/home/yoonji/AnatoMask/evaluate_uncertainty_totalsegmentator.py`: 통합 구현

### 데이터셋
- TotalSegmentator: 의료 이미지 자동 분할 데이터셋
- 경로: `/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/`

## 추가 확장 아이디어

### 1. 불확실성 기반 품질 보증

```python
# 높은 불확실성 영역 자동 감지
high_unc_mask = uncertainty > threshold
quality_score = 1.0 - (high_unc_mask.sum() / total_voxels)
```

### 2. 불확실성 기반 하드 케이스 마이닝

```python
# 높은 불확실성을 가진 샘플 우선 학습
hard_samples = [s for s in dataset if mean_uncertainty(s) > threshold]
```

### 3. 불확실성을 고려한 앙상블

```python
# 불확실성이 낮은 예측에 더 높은 가중치
weights = 1.0 / (uncertainty + epsilon)
ensemble_pred = (pred1 * w1 + pred2 * w2) / (w1 + w2)
```

## 라이선스 및 인용

이 코드는 WHAT 논문의 개념을 기반으로 하며, STUNetTrainer와 nnUNet을 활용합니다.

```bibtex
@article{kendall2017what,
  title={What uncertainties do we need in bayesian deep learning for computer vision?},
  author={Kendall, Alex and Gal, Yarin},
  journal={arXiv preprint arXiv:1506.02640},
  year={2017}
}
```

## 연락처

질문이나 버그 리포트는 [이메일] 또는 GitHub 이슈로 보내주세요.
