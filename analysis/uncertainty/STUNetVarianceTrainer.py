"""
STUNetVarianceTrainer: STUNetTrainer with Aleatoric Uncertainty Loss

Combines STUNet segmentation with per-location variance prediction
for uncertainty estimation.
"""

from nnunetv2.training.nnUNetTrainer.STUNetTrainer import STUNetTrainer
import torch
from torch import nn
from typing import Dict
import torch.nn.functional as F


class VarianceHead(nn.Module):
    """
    Variance prediction head for aleatoric uncertainty estimation.
    Takes segmentation logits and predicts log variance at each location.
    """
    def __init__(self, num_classes: int = 105):
        super().__init__()
        self.log_var_head = nn.Sequential(
            nn.Conv3d(num_classes, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W, D] segmentation logits

        Returns:
            log_var: [B, 1, H, W, D] log variance
        """
        return self.log_var_head(logits)


class CombinedSegmentationVarianceLoss(nn.Module):
    """
    Combined Loss for Segmentation + Aleatoric Uncertainty

    Loss = Dice Loss + λ * Variance Loss
    Variance Loss = 0.5 * [exp(-log_var) * (1 - target_prob)^2 + log_var]
    """

    def __init__(self, num_classes: int = 105, seg_weight: float = 1.0, var_weight: float = 0.5,
                 smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.seg_weight = seg_weight
        self.var_weight = var_weight
        self.smooth = smooth

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss for multi-class segmentation.

        Args:
            pred: [B, C, H, W, D] segmentation logits
            target: [B, H, W, D] class indices
        """
        # Safety check: ensure spatial dimensions match
        if pred.shape[2:] != target.shape[1:]:
            # Use nearest neighbor interpolation to avoid rounding errors
            target_np = target.unsqueeze(1).float()  # [B, 1, H, W, D]
            target = F.interpolate(
                target_np,
                size=pred.shape[2:],
                mode='nearest'  # Use nearest instead of trilinear to avoid size mismatches
            ).long().squeeze(1)  # [B, H', W', D']

        # Convert target to one-hot
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred, dim=1)

        # Compute dice for each class
        intersection = torch.sum(pred_probs * target_one_hot, dim=(2, 3, 4))
        union = torch.sum(pred_probs, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))

        # Avoid division by zero
        dice = 2 * intersection / (union + self.smooth)

        # Return mean dice loss
        return 1.0 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: [B, C, H, W, D] segmentation logits
            target: [B, H, W, D] or [B, 1, H, W, D] class indices
            log_var: [B, 1, H, W, D] log variance

        Returns:
            dict with 'total', 'seg', 'var' losses
        """
        # Ensure targets have correct shape
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)  # [B, H, W, D]
        elif target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # [B, H, W, D]

        # Now target should be [B, H, W, D]
        # Ensure size compatibility with pred [B, C, H', W', D']
        target_resized = target.long()
        if pred.shape[2:] != target_resized.shape[1:]:
            # Resize target to match pred spatial dimensions
            target_resized = F.interpolate(
                target_resized.unsqueeze(1).float(),  # [B, 1, H, W, D]
                size=pred.shape[2:],
                mode='trilinear',
                align_corners=False
            ).long().squeeze(1)  # [B, H', W', D']

        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        # ===== Segmentation Loss (Dice) =====
        seg_loss = self.dice_loss(pred, target_resized)

        # ===== Variance Loss (Aleatoric Uncertainty) =====
        # Compute softmax probabilities
        seg_probs = F.softmax(pred, dim=1)  # [B, C, H, W, D]

        # Get predicted probability of target class
        # target_resized is [B, H, W, D], need to unsqueeze to [B, 1, H, W, D] for gather
        target_idx = target_resized.unsqueeze(1).long()  # [B, 1, H, W, D]
        target_probs = torch.gather(
            seg_probs,
            dim=1,
            index=target_idx
        )  # [B, 1, H, W, D]

        # Uncertainty loss: allow high variance when prediction is uncertain
        uncertainty = 0.5 * (torch.exp(-log_var) * (1.0 - target_probs) ** 2 + log_var)
        var_loss = uncertainty.mean()

        # ===== Combined Loss =====
        total_loss = self.seg_weight * seg_loss + self.var_weight * var_loss

        return {
            'total': total_loss,
            'seg': seg_loss,
            'var': var_loss
        }


class STUNetVarianceTrainer(STUNetTrainer):
    """
    STUNetTrainer with Aleatoric Uncertainty Head

    Trains STUNet with both segmentation and variance prediction.
    """

    def __init__(self, *args, var_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_weight = var_weight
        self.variance_head = None
        self.combined_loss_fn = None

    def initialize(self):
        """Initialize trainer with variance head."""
        super().initialize()

        # Add variance head
        if self.network is not None:
            self.variance_head = VarianceHead(num_classes=self.label_manager.num_segmentation_heads)
            self.variance_head = self.variance_head.to(self.device)

            # Add variance head to DDP if needed
            if self.is_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.variance_head = DDP(self.variance_head, device_ids=[self.local_rank])

            # Replace loss function
            self.loss = CombinedSegmentationVarianceLoss(
                num_classes=self.label_manager.num_segmentation_heads,
                seg_weight=1.0,
                var_weight=self.var_weight
            )

    def forward(self, batch):
        """
        Forward pass with segmentation and variance prediction.

        Args:
            batch: dict with 'data' key

        Returns:
            tuple of (seg_logits, log_var)
        """
        data = batch['data']

        # Forward through STUNet
        if self.enable_deep_supervision:
            seg_logits = self.network(data)[0]  # Use highest resolution for variance
        else:
            seg_logits = self.network(data)

        # Forward through variance head
        log_var = self.variance_head(seg_logits)

        return seg_logits, log_var

    def training_step(self, batch):
        """
        Training step for one batch.

        Args:
            batch: dict with 'data' and 'target' keys

        Returns:
            loss tensor
        """
        data = batch['data']
        target = batch['target']

        # Forward through network
        output = self.network(data)

        # Handle deep supervision
        if self.enable_deep_supervision:
            seg_logits = output[0]  # Highest resolution
        else:
            seg_logits = output

        # Forward through variance head
        log_var = self.variance_head(seg_logits)

        # Compute combined loss
        loss_dict = self.loss(seg_logits, target, log_var)

        return loss_dict['total']

    def on_train_epoch_end(self):
        """Hook called at end of epoch."""
        super().on_train_epoch_end()
        # Can add custom logging for variance loss if needed

    def predict_with_variance(self, data: torch.Tensor) -> tuple:
        """
        Inference with uncertainty prediction.

        Args:
            data: [B, C, H, W, D] input tensor

        Returns:
            tuple of:
            - segmentation: [B, H, W, D] class indices
            - log_var: [B, 1, H, W, D] log variance
            - uncertainty: [B, 1, H, W, D] uncertainty map (variance)
            - confidence: [B, 1, H, W, D] confidence map (1 - variance)
        """
        self.network.eval()
        if self.variance_head is not None:
            self.variance_head.eval()

        with torch.no_grad():
            # Forward through STUNet
            output = self.network(data)

            # Handle deep supervision
            if self.enable_deep_supervision:
                seg_logits = output[0]
            else:
                seg_logits = output

            # Forward through variance head
            log_var = self.variance_head(seg_logits)

            # Extract segmentation
            segmentation = seg_logits.argmax(dim=1)  # [B, H, W, D]

            # Compute uncertainty and confidence
            uncertainty = torch.exp(log_var)  # [B, 1, H, W, D]
            confidence = 1.0 - uncertainty.clamp(max=1.0)  # [B, 1, H, W, D]

        return segmentation, log_var, uncertainty, confidence
