"""
STUNet Finetuning for Calibration on TotalSegmentator
Fine-tune STUNet model on TotalSegmentator training set for 20 epochs
to prepare for calibration curve analysis.
"""
import os

# 0번 GPU만 쓰고 싶을 때
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add paths
sys.path.insert(0, '/home/yoonji/AnatoMask/')
sys.path.insert(0, '/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain')

# Import nnUNet components
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

# Import STUNet
from nnunetv2.training.nnUNetTrainer.STUNetTrainer import STUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =============================================================================
# Configuration
# =============================================================================
# PRETRAINED_MODEL = "/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrained_model/large_ep4k.model"
PRETRAINED_MODEL = "/home/yoonji/MedMIM-1/stunet_finetuned_for_calibration/checkpoint_epoch_14.pth"
OUTPUT_DIR = "/home/yoonji/MedMIM-1/stunet_finetuned_for_calibration"
DATASET_NAME = "Dataset606_all_TotalSegmentator"

# Dataset paths
PREPROCESSED_FOLDER = f"/nas_homes/yoonji/medmask/nnUNet_preprocessed/{DATASET_NAME}/nnUNetPlans_3d_fullres"
SPLITS_FILE = f"/nas_homes/yoonji/medmask/nnUNet_preprocessed/{DATASET_NAME}/splits_final.json"
DATASET_JSON = f"/nas_homes/yoonji/medmask/nnUNet_preprocessed/{DATASET_NAME}/dataset.json"
PLANS_JSON = f"/nas_homes/yoonji/medmask/nnUNet_preprocessed/{DATASET_NAME}/nnUNetPlans.json"

# Training hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 1  # Reduced to avoid OOM on 48GB GPU
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 2 via grad accumulation
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
FOLD = 0

# Model configuration
pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

# =============================================================================
# Data Transforms
# =============================================================================
def get_training_transforms(deep_supervision_scales):
    tr_transforms = []
    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(RenameTransform('seg', 'target', True))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0,
                                                          input_key='target', output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    return Compose(tr_transforms)

# =============================================================================
# Dice + CE Loss
# =============================================================================
class DiceCELoss(nn.Module):
    def __init__(self, num_classes, weight_ce=0.5, weight_dice=0.5, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if target.dim() == 5 and target.shape[1] == 1:
            target_proc = target.squeeze(1)
        else:
            target_proc = target

        # ensure target spatial size matches prediction
        if target_proc.shape[-3:] != pred.shape[2:]:
            target_proc = F.interpolate(
                target_proc.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1)

        target_long = target_proc.long()

        ce = self.ce_loss(pred, target_long)
        dice = self._dice_loss(pred, target_long)

        return self.weight_ce * ce + self.weight_dice * dice

    def _dice_loss(self, pred, target):
        probs = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs, dims) + torch.sum(target_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return dice_loss

# =============================================================================
# Dice Score
# =============================================================================
def dice_score(pred, target, num_classes, eps=1e-8):
    """Compute Dice on CPU to reduce GPU memory pressure."""
    with torch.no_grad():
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        if pred.shape[2:] != target.shape[-3:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1)

        target_cpu = target.long().cpu()

        # Argmax on GPU, then move to CPU (int16 to save RAM)
        pred_labels = pred.argmax(dim=1).to(torch.int16).cpu()

        dice_scores = []
        for c in range(1, num_classes):  # Skip background
            pred_c = (pred_labels == c)
            target_c = (target_cpu == c)

            inter = (pred_c & target_c).sum().item()
            union = pred_c.sum().item() + target_c.sum().item()

            if union > 0:
                dice = (2.0 * inter + float(eps)) / (union + float(eps))
                dice_scores.append(dice)

        if dice_scores:
            dice_mean = sum(dice_scores) / len(dice_scores)
        else:
            dice_mean = 0.0

    return pred.new_tensor(dice_mean)

# =============================================================================
# Main Training
# =============================================================================
def main():
    print("="*80)
    print("STUNet Finetuning for Calibration")
    print("="*80)
    print(f"Pretrained Model: {PRETRAINED_MODEL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*80)

    # Create output directory
    maybe_mkdir_p(OUTPUT_DIR)

    # Load dataset configuration
    print("\nLoading dataset configuration...")
    splits = load_json(SPLITS_FILE)
    dataset_json = load_json(DATASET_JSON)
    plans = load_json(PLANS_JSON)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    label_manager = plans_manager.get_label_manager(dataset_json)

    num_classes = len(label_manager.all_labels)
    print(f"Number of classes: {num_classes}")

    # Get training keys (only train split, no validation for finetuning)
    train_keys = splits[FOLD]['train']
    print(f"Training samples: {len(train_keys)}")

    # Create dataset
    dataset_tr = nnUNetDataset(PREPROCESSED_FOLDER, train_keys,
                               folder_with_segs_from_previous_stage=None,
                               num_images_properties_loading_threshold=0)

    # Data loader configuration
    # patch_size = configuration_manager.patch_size
    patch_size = (112, 112, 128)  # Reduced from (160,160,160) to avoid OOM
    rotation_for_DA = {
        'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    }
    initial_patch_size = get_patch_size(patch_size, *rotation_for_DA.values(), (0.85, 1.25))

    # Disable deep supervision for simplicity
    deep_supervision_scales = None

    # Get transforms
    tr_transforms = get_training_transforms(deep_supervision_scales)

    # Create dataloader
    dl_tr = nnUNetDataLoader3D(dataset_tr, BATCH_SIZE,
                               initial_patch_size, patch_size, label_manager,
                               oversample_foreground_percent=0.5,
                               sampling_probabilities=None, pad_sides=None)

    iters_train = math.ceil(len(dataset_tr) / BATCH_SIZE)
    allowed_num_processes = get_allowed_n_proc_DA()

    # Initialize model
    print("\nInitializing STUNet model...")
    model = STUNet(
        input_channels=1,
        num_classes=num_classes,
        depth=[1, 1, 1, 1, 1, 1],
        dims=[32, 64, 128, 256, 512, 512],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        enable_deep_supervision=False
    )

    # Load pretrained weights
    print(f"Loading pretrained weights from {PRETRAINED_MODEL}...")
    pretrained_weights = torch.load(PRETRAINED_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(pretrained_weights, strict=False)
    print("Pretrained weights loaded successfully!")

    model = model.to(device)
    model.train()

    # 1) 전부 일단 freeze
    for p in model.parameters():
        p.requires_grad = False

    # 2) encoder만 명시적으로 freeze (가독성용)
    for p in model.conv_blocks_context.parameters():
        p.requires_grad = False

    # 3) decoder + seg_outputs는 trainable
    for p in model.upsample_layers.parameters():
        p.requires_grad = True

    for p in model.conv_blocks_localization.parameters():
        p.requires_grad = True

    for p in model.seg_outputs.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = DiceCELoss(num_classes=num_classes, weight_ce=0.5, weight_dice=0.5)
    scaler = GradScaler(enabled=True)

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        num_batches = 0

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

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

        optimizer.zero_grad(set_to_none=True)

        for batch_idx in range(iters_train):
            # Get batch
            batch = next(mt_gen_train)
            inp = batch['data'].to(device, non_blocking=True)
            target = batch['target']

            if isinstance(target, (list, tuple)):
                target = target[0].to(device, non_blocking=True)
            else:
                target = target.to(device, non_blocking=True)

            with autocast(enabled=True):
                pred = model(inp)
                loss_raw = criterion(pred, target)

            loss_value = loss_raw.item()
            loss = loss_raw / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            should_step = ((batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == iters_train)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Compute metrics
            with torch.no_grad():
                if isinstance(pred, (list, tuple)):
                    pred_for_dice = pred[0]
                else:
                    pred_for_dice = pred
                dice = dice_score(pred_for_dice, target, num_classes)

            epoch_loss += loss_value
            epoch_dice += dice.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{iters_train}] - "
                      f"Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Dice: {avg_dice:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'dice': avg_dice,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dice': avg_dice,
            }, best_path)
            print(f"  Best model saved: {best_path}")

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Final model saved: {final_path}")
    print(f"Best model saved: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
    print("="*80)

if __name__ == '__main__':
    main()
