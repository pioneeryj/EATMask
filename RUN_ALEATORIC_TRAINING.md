# STUNet Aleatoric Uncertainty Training Guide

## 개요

이 가이드는 STUNet 모델을 사용하여 WHAT 논문의 **Aleatoric Uncertainty**를 TotalSegmentator 데이터셋에서 학습하는 방법을 설명합니다.

## 핵심 개념

### Aleatoric Uncertainty Loss

```
Loss = 0.5 * [exp(-σ²) * (mean - target)² + σ²]
```

**의미:**
- `σ²`: 각 복셀에서의 불확실성 (variance)
- 높은 σ²: 모델이 그 영역에서 불확실함
- 낮은 σ²: 모델이 확신함

### 아키텍처

```
입력 이미지 [B, 1, 112, 112, 128]
    ↓
STUNet (기존 분할 모델)
    ↓
두 개의 출력:
  - mean:  분할 예측 [B, 1, 112, 112, 128]
  - var:   불확실성   [B, 1, 112, 112, 128]
    ↓
Loss = AleatoricUncertaintyLoss(mean, var, target)
```

## 파일 구조

```
/home/yoonji/AnatoMask/
├── train_stunet_aleatoric.py              # 메인 학습 스크립트
├── nnunetv2/training/loss/
│   └── aleatoric_uncertainty_loss.py      # 손실함수
└── Anatomask_results/aleatoric_uncertainty/  # 결과 저장 폴더
```

## 실행 방법

### 1. 단일 GPU 학습

```bash
cd /home/yoonji/AnatoMask

python train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --warmup 20 \
    --model_name stunet_aleatoric
```

### 2. 다중 GPU 학습 (DDP)

```bash
cd /home/yoonji/AnatoMask

# 4개 GPU에서 실행
torchrun --nproc_per_node=4 train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --warmup 20 \
    --model_name stunet_aleatoric
```

### 3. 특정 GPU 선택

```bash
# GPU 0, 1, 2, 3 사용
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 4
```

### 4. 커스텀 출력 폴더

```bash
python train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 4 \
    --output /custom/output/path
```

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--fold` | 0 | 학습할 폴드 (0-4) |
| `--epochs` | 100 | 에포크 수 |
| `--batch_size` | 4 | 배치 크기 (전체 크기) |
| `--lr` | 1e-4 | 학습률 |
| `--warmup` | 20 | Warmup 에포크 수 |
| `--model_name` | stunet_aleatoric | 모델 이름 |
| `--output` | auto | 출력 폴더 (미지정시 자동) |

## 예제

### 예제 1: 기본 학습

```bash
python train_stunet_aleatoric.py --fold 0 --epochs 50 --batch_size 4
```

**결과:**
```
Epoch 0/50
Training: 100%|██████████| 50/50 [00:30<00:00,  1.66it/s]
  loss: 0.2345, var: 0.4567
✓ checkpoint saved
...
Epoch 49/50
✓ Training completed!
```

### 예제 2: 더 많은 에포크로 학습

```bash
python train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 200 \
    --batch_size 2 \
    --lr 5e-5
```

### 예제 3: 다중 폴드 학습

```bash
# 모든 폴드 학습
for fold in {0..4}; do
    python train_stunet_aleatoric.py \
        --fold $fold \
        --epochs 100 \
        --batch_size 4
done
```

### 예제 4: 분산 학습 (4 GPU)

```bash
torchrun --nproc_per_node=4 train_stunet_aleatoric.py \
    --fold 0 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4
```

## 출력 및 모니터링

### 터미널 출력

```
📚 Loading data...
✓ Training samples: 850
✓ Validation samples: 150

⚙️  Setting up data loaders...
✓ Batch size per GPU: 4

🧠 Building STUNet...
✓ Model created with 1,234,567 parameters

⚙️  Setting up training...

======================================================================
STARTING STUNET ALEATORIC UNCERTAINTY TRAINING
======================================================================

Epoch 0/100
Training: 100%|██████████| 50/50 [00:35<00:00,  1.43it/s]
  loss: 0.2345, var: 0.4567

Epoch 0 AVG Loss: 0.234567
Epoch 0 Var Mean: 0.456789

...
```

### 로그 파일

```
/home/yoonji/AnatoMask/Anatomask_results/aleatoric_uncertainty/stunet_aleatoric/fold0/
├── training_log_2024_12_02_14_30_45.txt  # 상세 로그
├── stunet_aleatoric_epoch_000.pt          # 에포크 체크포인트
├── stunet_aleatoric_epoch_010.pt
└── stunet_aleatoric_best.pt               # 최고 성능 모델
```

## 체크포인트 로드

### 학습된 모델 사용

```python
import torch
from train_stunet_aleatoric import STUNetWithUncertainty
from STUNet_head import STUNet

# 체크포인트 로드
checkpoint = torch.load('/path/to/stunet_aleatoric_best.pt')

# 모델 재구성
head = STUNet(1, 1, depth=[1, 1, 1, 1, 1, 1],
              dims=[32, 64, 128, 256, 512, 512])
model = STUNetWithUncertainty(head)

# 가중치 로드
model.load_state_dict(checkpoint['network_weights'])
model = model.cuda()

# 불확실성 추정
with torch.no_grad():
    image = torch.randn(1, 1, 112, 112, 128).cuda()
    output = model(image)
    mean = output['mean']        # 분할 예측
    uncertainty = output['var']  # 알레아토릭 불확실성
```

## 손실함수 상세 설명

### WHAT 논문의 Loss

```
L = 0.5 * [e^(-σ²) * (ŷ - y)² + σ²]
```

**해석:**

1. **첫 번째 항**: `e^(-σ²) * (ŷ - y)²`
   - σ²가 작음 → 큰 MSE 페널티
   - σ²가 큼 → 작은 MSE 페널티
   - 모델이 작은 분산을 예측할 때만 정확한 예측을 요구

2. **두 번째 항**: `σ²`
   - 분산이 무한정 커지는 것을 방지
   - 분산을 적절한 수준으로 유지

3. **결과**:
   - 예측이 어려운 영역 (경계 등): σ² 증가
   - 예측이 쉬운 영역 (배경 등): σ² 감소
   - 모델이 자동으로 영역별 불확실성을 학습

### 정규화

학습 과정에서:
```python
var_reg = 0.01 * (torch.log(var.mean() + 1e-6)) ** 2
total_loss = loss + var_reg
```

이를 통해:
- Variance가 극단적으로 커지지 않도록 제어
- 수치 안정성 확보

## 성능 최적화

### 메모리 사용 최소화

```bash
# 배치 크기 감소
python train_stunet_aleatoric.py --batch_size 2

# 또는 패치 크기 감소 (코드 수정 필요)
# self.patch_size = [96, 96, 96]
```

### 학습 속도 향상

```bash
# 다중 GPU 사용
torchrun --nproc_per_node=4 train_stunet_aleatoric.py --batch_size 8

# 또는 학습률 증가 (수렴이 빨라질 수 있음)
python train_stunet_aleatoric.py --lr 2e-4
```

## 트러블슈팅

### 문제 1: CUDA out of memory

**해결책:**
```bash
# 배치 크기 감소
python train_stunet_aleatoric.py --batch_size 2
```

### 문제 2: 손실값이 NaN

**원인:** 분산이 0이 되어 log(0) 계산

**해결책:**
```python
# epsilon 추가
var = torch.clamp(var, min=1e-6)
```

### 문제 3: 분산이 모두 같음

**원인:** Variance head가 제대로 학습되지 않음

**해결책:**
```python
# Variance head 초기화 확인
# 또는 learning rate 조정
python train_stunet_aleatoric.py --lr 5e-5
```

### 문제 4: DDP 에러

**해러 메시지:** "RuntimeError: Expected to have finished reduction in the prior iteration"

**해결책:**
```python
# 모델 구성 시 find_unused_parameters=True 확인
model = DDP(model_without_ddp, device_ids=[local_rank],
            find_unused_parameters=True, broadcast_buffers=False)
```

## 결과 분석

### 메트릭 해석

```
Epoch 50 AVG Loss: 0.234567    # 낮을수록 좋음
Epoch 50 Var Mean: 0.456789    # 보통 0.1 ~ 1.0
```

**Loss가 높은 경우:**
- 학습이 덜 진행됨
- 더 많은 에포크 필요
- 학습률 조정 필요

**Var Mean이 0에 가까운 경우:**
- 모델이 너무 확신함
- 과적합 가능성
- Variance head의 가중치 조정 필요

**Var Mean이 매우 큰 경우 (> 2.0):**
- 모델이 모든 곳에서 불확실함
- 학습이 덜 진행됨
- 학습률 증가 또는 에포크 수 증가

## 고급 사용법

### 커스텀 학습률 스케줄

코드 수정:
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)
```

### 앙상블 학습

```bash
# 5개 폴드 모두 학습
for fold in {0..4}; do
    python train_stunet_aleatoric.py --fold $fold --epochs 100
done
```

### 재개 학습 (체크포인트에서)

```python
# 코드 수정 필요: checkpoint 로드 및 resume 옵션 추가
checkpoint = torch.load('/path/to/epoch_050.pt')
model.load_state_dict(checkpoint['network_weights'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_epoch = checkpoint['epoch'] + 1
```

## 참고 자료

### WHAT 논문
```bibtex
@article{kendall2017what,
  title={What uncertainties do we need in bayesian deep learning for computer vision?},
  author={Kendall, Alex and Gal, Yarin},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={39},
  number={9},
  pages={1848--1862},
  year={2017}
}
```

### 관련 파일
- STUNet: `/home/yoonji/AnatoMask/STUNet_head.py`
- Loss: `/home/yoonji/AnatoMask/nnunetv2/training/loss/aleatoric_uncertainty_loss.py`
- 기존 pretrain: `/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrain_MedMask.py`

## FAQ

**Q: 배치 크기는 얼마나 해야 하나?**
A: 기본값 4가 좋습니다. GPU 메모리가 충분하면 8로 증가 가능합니다.

**Q: 에포크 수는?**
A: 100 에포크를 권장합니다. Loss가 더 감소하면 더 많이 해도 됩니다.

**Q: 학습률은?**
A: 기본값 1e-4가 좋습니다. Loss가 수렴하지 않으면 조정하세요.

**Q: DDP를 꼭 써야 하나?**
A: 아닙니다. 단일 GPU로도 학습 가능합니다 (느리지만).

**Q: 불확실성 맵은 어디에 저장되나?**
A: 현재는 모델 체크포인트에만 저장됩니다. 후처리로 추출 가능합니다.

---

**마지막 업데이트**: 2024년 12월
**버전**: 1.0
**Contact**: [이메일 또는 연락처]
