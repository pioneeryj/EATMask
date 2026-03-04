# Aleatoric Uncertainty Training - Quick Start

## TL;DR (30초 요약)

STUNet으로 알레아토릭 불확실성을 학습하세요:

```bash
python train_stunet_aleatoric.py --fold 0 --epochs 100 --batch_size 4
```

## 파일 설명

### 1. `train_stunet_aleatoric.py` ⭐ (메인 파일)
- **목적**: STUNet으로 Aleatoric Uncertainty 학습
- **특징**:
  - WHAT 논문의 Loss 함수 구현
  - DDP 지원 (다중 GPU)
  - TotalSegmentator 데이터 사용

### 2. `nnunetv2/training/loss/aleatoric_uncertainty_loss.py`
- **목적**: 손실함수 모음
- **포함**: AleatoricUncertaintyLoss, NegativeLogLikelihoodLoss 등

### 3. `RUN_ALEATORIC_TRAINING.md`
- **목적**: 상세 가이드
- **포함**: 하이퍼파라미터, 트러블슈팅, 고급 사용법

## 핵심 개념

### 손실함수

```
Loss = 0.5 * [exp(-σ²) * (mean - target)² + σ²]
```

- **mean**: 분할 예측
- **σ²**: 각 복셀에서의 불확실성
- 자동으로 각 영역의 적절한 불확실성을 학습합니다

### 모델 구조

```
STUNet (분할 모델)
  ├─ Mean Head  → 분할 예측
  └─ Variance Head → 불확실성
```

## 실행 방법

### 최소 실행

```bash
cd /home/yoonji/AnatoMask
python train_stunet_aleatoric.py
```

### 권장 실행

```bash
python train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --warmup 20 \
    --model_name stunet_aleatoric
```

### 다중 GPU 실행

```bash
torchrun --nproc_per_node=4 train_stunet_aleatoric.py \
    --fold 0 --epochs 100 --batch_size 4
```

## 주요 파라미터

| 파라미터 | 기본값 | 권장값 |
|---------|-------|--------|
| `--fold` | 0 | 0-4 |
| `--epochs` | 100 | 50-200 |
| `--batch_size` | 4 | 2-8 |
| `--lr` | 1e-4 | 5e-5 ~ 5e-4 |
| `--warmup` | 20 | 10-30 |

## 학습 모니터링

### 터미널 출력

```
Epoch 50/100
Training: 100%|██████████| 50/50 [00:30]
  loss: 0.2345, var: 0.4567
✓ checkpoint saved
```

### 메트릭 해석

- **loss**: 낮을수록 좋음 (목표: < 0.3)
- **var**: 평균 불확실성 (정상: 0.1 ~ 1.0)

## 결과 저장 위치

```
/home/yoonji/AnatoMask/Anatomask_results/aleatoric_uncertainty/
└── stunet_aleatoric/
    └── fold0/
        ├── training_log_*.txt          # 학습 로그
        ├── stunet_aleatoric_best.pt    # 최고 성능 모델
        └── stunet_aleatoric_epoch_*.pt # 에포크별 체크포인트
```

## 학습된 모델 사용

```python
import torch
from train_stunet_aleatoric import STUNetWithUncertainty
from STUNet_head import STUNet

# 모델 로드
checkpoint = torch.load('/path/to/stunet_aleatoric_best.pt')
head = STUNet(1, 1, depth=[1, 1, 1, 1, 1, 1],
              dims=[32, 64, 128, 256, 512, 512])
model = STUNetWithUncertainty(head)
model.load_state_dict(checkpoint['network_weights'])
model.cuda()

# 추론
with torch.no_grad():
    image = torch.randn(1, 1, 112, 112, 128).cuda()
    output = model(image)
    mean = output['mean']        # 분할
    uncertainty = output['var']  # 불확실성
```

## 자주하는 실수

### ❌ 실수 1: 데이터 경로 확인 안 함
```python
# 확인 필요:
preprocessed_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans_3d_fullres'
```

### ❌ 실수 2: 배치 크기 너무 크게
```bash
# 나쁜 예
python train_stunet_aleatoric.py --batch_size 32

# 좋은 예
python train_stunet_aleatoric.py --batch_size 4
```

### ❌ 실수 3: 에포크 너무 적게
```bash
# 나쁜 예
python train_stunet_aleatoric.py --epochs 10

# 좋은 예
python train_stunet_aleatoric.py --epochs 100
```

## 트러블슈팅 (자세한 내용은 RUN_ALEATORIC_TRAINING.md 참조)

| 문제 | 해결책 |
|------|--------|
| CUDA out of memory | `--batch_size 2` 또는 `--batch_size 1` |
| Loss가 NaN | 배치 크기 감소 또는 학습률 감소 |
| 학습이 진행 안 됨 | `--epochs 200` 또는 `--lr 5e-5` |
| DDP 에러 | 코드의 `find_unused_parameters=True` 확인 |

## 경과 시간

| 구성 | 에포크 | 예상 시간 |
|------|--------|---------|
| 단일 GPU | 100 | ~5-6시간 |
| 4 GPU (DDP) | 100 | ~1.5-2시간 |
| 단일 GPU | 200 | ~10-12시간 |

## 다음 단계

1. **학습** (현재)
   ```bash
   python train_stunet_aleatoric.py --fold 0 --epochs 100
   ```

2. **불확실성 평가** (학습 후)
   ```bash
   python evaluate_uncertainty_totalsegmentator.py \
       --model_path /path/to/stunet_aleatoric_best.pt
   ```

3. **시각화** (평가 후)
   ```bash
   python visualize_uncertainty_maps.py \
       --output_dir /tmp/uncertainty_viz
   ```

## 참고 자료

- **논문**: [WHAT - What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1506.02640)
- **상세 가이드**: `RUN_ALEATORIC_TRAINING.md`
- **손실함수**: `nnunetv2/training/loss/aleatoric_uncertainty_loss.py`
- **기존 pretrain**: `nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrain_MedMask.py`

## 간단한 예제

### 최소 실행 코드

```bash
# 1. 디렉토리 이동
cd /home/yoonji/AnatoMask

# 2. 학습 시작 (기본 설정)
python train_stunet_aleatoric.py --fold 0

# 3. 완료!
# 결과: /home/yoonji/AnatoMask/Anatomask_results/aleatoric_uncertainty/
```

### 커스텀 실행 코드

```bash
# 모든 폴드에서 학습
for fold in {0..4}; do
    echo "Training fold $fold..."
    python train_stunet_aleatoric.py \
        --fold $fold \
        --epochs 100 \
        --batch_size 4
done
```

### 다중 GPU 실행

```bash
# 4개 GPU에서 분산 학습
torchrun --nproc_per_node=4 train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 8
```

## 성능 기대값

- **Loss**: 0.2 ~ 0.3 (수렴 후)
- **Variance Mean**: 0.3 ~ 0.8 (학습된 모델)
- **수렴 시간**: 50-100 에포크

## 체크리스트

- [ ] 데이터 경로 확인
- [ ] GPU 메모리 확인 (최소 8GB)
- [ ] 파이썬 패키지 설치 확인
- [ ] 첫 실행 (작은 에포크로 테스트)
- [ ] 전체 학습 실행
- [ ] 결과 확인
- [ ] 모델 저장

## 질문이 있으신가요?

1. **RUN_ALEATORIC_TRAINING.md** - 상세 가이드
2. **코드 주석** - `train_stunet_aleatoric.py` 참조
3. **원본 코드** - `/home/yoonji/AnatoMask/what/WHAT_src/` 참조

---

**준비되셨나요? 이제 시작하세요!**

```bash
python train_stunet_aleatoric.py --fold 0 --epochs 100
```

**Happy Training! 🚀**
