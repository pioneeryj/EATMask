"""
Aleatoric Uncertainty Loss (WHAT 논문 기반)

WHAT: What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
https://arxiv.org/abs/1506.02640

핵심:
- 모델이 예측값(mean)과 불확실성(variance)을 동시에 학습
- Loss = 0.5 * [exp(-var) * (mean - target)^2 + var]
- 높은 variance: 모델이 그 영역에서 불확실함
- 낮은 variance: 모델이 자신감 있음
"""

import torch
import torch.nn as nn


class AleatoricUncertaintyLoss(nn.Module):
    """
    Aleatoric Uncertainty Loss 함수

    모델 출력: {'mean': prediction, 'var': uncertainty_variance}
    """

    def __init__(self, var_weight: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            var_weight: variance의 가중치 (기본값: 1.0)
                       높을수록 variance 학습에 더 강조
            reduction: 손실 계산 방식
                      'mean': 평균 손실
                      'sum': 합 손실
                      'none': 요소별 손실
        """
        super().__init__()
        self.var_weight = var_weight
        self.reduction = reduction

    def forward(self,
                predictions: dict,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: 모델 출력
                - 'mean': 예측값 [B, C, H, W, D]
                - 'var': 분산값 [B, C, H, W, D]
            targets: 타겟 [B, C, H, W, D]

        Returns:
            scalar loss 값
        """
        mean = predictions['mean']
        var = predictions['var']

        # Variance 가중치 적용
        var = self.var_weight * var

        # Loss 계산: 0.5 * [exp(-var) * (mean - target)^2 + var]
        # 이는 다음과 동치: -0.5 * log(var) - 0.5 * (mean-target)^2/var + 0.5 * var
        # (정규화 상수 제외)

        # exp(-var) * (mean - target)^2 + var
        mse_loss = torch.exp(-var) * (mean - targets) ** 2
        var_loss = var

        loss = 0.5 * (mse_loss + var_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class NegativeLogLikelihoodLoss(nn.Module):
    """
    음수 최대우도 손실함수 (Negative Log Likelihood)

    heteroscedastic 모델에 더 적합한 형태:
    NLL = 0.5 * [log(var) + (mean - target)^2 / var]
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.reduction = reduction
        self.epsilon = 1e-6  # numerical stability

    def forward(self,
                predictions: dict,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: {'mean': ..., 'var': ...}
            targets: target values

        Returns:
            scalar loss
        """
        mean = predictions['mean']
        var = predictions['var']

        # Numerical stability
        var = torch.clamp(var, min=self.epsilon)

        # NLL Loss
        nll = 0.5 * (torch.log(var) + (mean - targets) ** 2 / var)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class WeightedAleatoricLoss(nn.Module):
    """
    가중치가 있는 Aleatoric Uncertainty Loss

    각 샘플 또는 영역에 다른 가중치를 적용할 수 있음
    예: 전경과 배경에 다른 가중치
    """

    def __init__(self, var_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.var_weight = var_weight
        self.reduction = reduction

    def forward(self,
                predictions: dict,
                targets: torch.Tensor,
                weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            predictions: {'mean': ..., 'var': ...}
            targets: target values
            weights: 각 요소의 가중치 [B, C, H, W, D] or None

        Returns:
            scalar loss
        """
        mean = predictions['mean']
        var = predictions['var']

        var = self.var_weight * var

        # 기본 손실
        mse_loss = torch.exp(-var) * (mean - targets) ** 2
        var_loss = var
        loss = 0.5 * (mse_loss + var_loss)

        # 가중치 적용
        if weights is not None:
            loss = loss * weights

        if self.reduction == 'mean':
            if weights is not None:
                return (loss.sum() / weights.sum().clamp(min=1.0))
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedAleatoricEpistemicLoss(nn.Module):
    """
    Aleatoric + Epistemic Uncertainty 결합 손실함수

    Epistemic: MC Dropout으로 추정 (출력 분산)
    Aleatoric: 직접 학습 (variance head)
    """

    def __init__(self,
                 aleatoric_weight: float = 1.0,
                 epistemichidden_weight: float = 0.1):
        """
        Args:
            aleatoric_weight: aleatoric loss의 가중치
            epistemic_weight: epistemic loss의 가중치
        """
        super().__init__()
        self.aleatoric_weight = aleatoric_weight
        self.epistemic_weight = epistemic_weight
        self.aleatoric_loss = AleatoricUncertaintyLoss(var_weight=1.0)

    def forward(self,
                predictions_list: list,  # MC dropout의 여러 출력
                predictions_with_var: dict,  # mean과 var
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions_list: MC dropout의 T개 출력 리스트
            predictions_with_var: {'mean': ..., 'var': ...}
            targets: target values

        Returns:
            scalar loss
        """
        # Aleatoric loss
        aleatoric_loss = self.aleatoric_loss(predictions_with_var, targets)

        # Epistemic loss (MC dropout 출력의 분산)
        predictions_stacked = torch.stack(predictions_list)  # [T, B, C, H, W, D]
        epistemic_var = predictions_stacked.var(dim=0)  # [B, C, H, W, D]

        # Epistemic variance를 최소화 (모델 확신도 증가)
        epistemic_loss = epistemic_var.mean()

        # 결합
        total_loss = (self.aleatoric_weight * aleatoric_loss +
                     self.epistemic_weight * epistemic_loss)

        return total_loss
