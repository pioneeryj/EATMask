# region Description of the region
import torch
device = torch.device("cuda:0")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
import time
from time import sleep
from datetime import datetime
import numpy as np
from timm.utils import ModelEma
import sys
sys.path.insert(0, '/home/yoonji/AnatoMask/')
sys.path.insert(0, '/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain')
from nnunetv2.training.lr_scheduler.LinearWarmupCosine import LinearWarmupCosineAnnealingLR
from nnunetv2.training.nnUNetTrainer.variants.pretrain.STUNet_head import STUNet
from nnunetv2.training.nnUNetTrainer.variants.pretrain.STUNet_dropout import STUNet_dropout


from nnunetv2.training.nnUNetTrainer.variants.pretrain.AnatoMask import SparK, monte_carlo, calculate_conditional_entropy,calculate_epistemic_uncertainty, calculate_aleatoric_uncertainty

from torch.cuda.amp import GradScaler, autocast
import sys
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from typing import Union, Tuple, List

import math
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from utils.lr_control import lr_wd_annealing, get_param_groups
from utils import dist

import torch.nn as nn
import torch.nn.functional as F
# from projection_head import ProjectionHead
import os

def get_validation_transforms(deep_supervision_scales: Union[List, Tuple],
                              is_cascaded: bool = False,
                              foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                              regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                              ignore_label: int = None) -> AbstractTransform:
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))

    if is_cascaded:
        val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        # the ignore label must also be converted
        val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                    if ignore_label is not None else regions,
                                                                    'target', 'target'))

    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                           output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)
    return val_transforms

# endregion

# Define your models here:
pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

# STUNet_B
head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
               enable_deep_supervision=True).to(device)
# STUNet_L
# from GC import STUNet
# head = STUNet(1,1,depth=[2] * 6, dims=[64 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
#               enable_deep_supervision=True).to(device)
# STUNet_H
# head = STUNet(1,1,depth=[3] * 6, dims=[96 * x for x in [1, 2, 4, 8, 16, 16]], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
#             enable_deep_supervision=True).to(device)

model_name = 'medmask'


### Your preprocessed dataset folder
preprocessed_dataset_folder = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans_3d_fullres'
#preprocessed_dataset_folder = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans_3d_fullres'
### Your nnUNet splits json
splits_file = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/splits_final.json'
splits = load_json(splits_file)
fold=0
all_keys = splits[fold]['train']
tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

dataset_tr = nnUNetDataset(preprocessed_dataset_folder, tr_keys,
                           folder_with_segs_from_previous_stage=None,
                           num_images_properties_loading_threshold=0)
dataset_val = nnUNetDataset(preprocessed_dataset_folder, val_keys,
                            folder_with_segs_from_previous_stage=None,
                            num_images_properties_loading_threshold=0)
### Your nnUNet dataset json
dataset_json =load_json('/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json')
### Your nnUNet plans json
plans = load_json('/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json')
plans_manager = PlansManager(plans)
### Your configurations
configuration_manager = plans_manager.get_configuration('3d_fullres')
label_manager = plans_manager.get_label_manager(dataset_json)

# patch_size = configuration_manager.patch_size
patch_size = [112,112, 128]
dim = len(patch_size)
rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
initial_patch_size = get_patch_size(patch_size[-dim:],
                                    *rotation_for_DA.values(),
                                    (0.85, 1.25))
iters_train = len(dataset_val)

deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            pool_op_kernel_sizes), axis=0))[:-1]
mirror_axes = (0, 1, 2)

val_transforms = get_validation_transforms(
    deep_supervision_scales,
    is_cascaded=False,
    foreground_labels=label_manager.foreground_labels,
    regions=label_manager.foreground_regions if
    label_manager.has_regions else None,
    ignore_label=label_manager.ignore_label)


allowed_num_processes = get_allowed_n_proc_DA()

dl_val = nnUNetDataLoader3D(dataset_val, 1,
                           initial_patch_size,
                           patch_size,
                           label_manager,
                           oversample_foreground_percent=0.33,
                           sampling_probabilities=None, pad_sides=None)


mt_gen_val = LimitedLenWrapper(iters_train, data_loader=dl_val, transform=val_transforms,
                                 num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                 pin_memory= True, wait_time=0.02)



# STUNet model with dropout for uncertainty estimation
model_dropout = STUNet_dropout(
    input_channels=1,
    num_classes=16,
    depth=[1, 1, 1, 1, 1, 1],
    dims=[32, 64, 128, 256, 512, 512],
    pool_op_kernel_sizes=pool_op_kernel_sizes,
    conv_kernel_sizes=conv_kernel_sizes,
    enable_deep_supervision=False,
    dropout_ratio=0
)
pretrained_model = "/nas_homes/yoonji/medmask/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/STUNetTrainer__nnUNetPlans__3d_fullres/spark_1000epoch/checkpoint_best.pth"  # pretrained weights 경로

pretrained_weights = torch.load(pretrained_model, map_location = device, weights_only=False)
model_dropout.load_state_dict(pretrained_weights, strict=False)
model_dropout = model_dropout.to(device)
model_dropout.eval()

from sklearn.calibration import calibration_curve

# Get number of classes dynamically from the model (not hardcoded)
num_classes = model_dropout.num_classes

# Store predictions and labels for each class
all_probs_per_class = [[] for _ in range(num_classes)]
all_labels_per_class = [[] for _ in range(num_classes)]

print(f"Number of classes for calibration: {num_classes}")
print(f"Dataset has {label_manager.num_segmentation_heads} classes")
print(f"Model outputs {num_classes} classes")

# Limit number of samples for memory efficiency
max_calibration_samples = min(iters_train, 50)  # Use max 50 samples
print(f"Processing {max_calibration_samples} validation samples (limited for memory efficiency)...")

try:
    for idx in range(max_calibration_samples):
        if idx % 10 == 0:
            print(f"Processing sample {idx+1}/{max_calibration_samples}")

        try:
            batch = next(mt_gen_val)
            inp = batch['data']
            target = batch['target']

            inp = inp.to(device, non_blocking=True)
            # Ground truth segmentation may be a list (deep supervision). Move accordingly.
            if isinstance(target, (list, tuple)):
                target = [t.to(device, non_blocking=True) for t in target]
            else:
                target = target.to(device, non_blocking=True)

            # Monte Carlo dropout for uncertainty estimation
            with torch.no_grad():
                logits = model_dropout(inp)

            # Handle deep supervision output - extract highest resolution
            if isinstance(logits, (list, tuple)):
                logits = logits[0]  # Get highest resolution output (B, C, H, W, D)
                print(logits.shape)

            # Calculate mean probabilities
            print(logits.shape)
            mean_softmax = torch.softmax(logits, dim=1)  # Shape: (B, C, H, W, D)

            # For each class, collect probabilities and binary labels
            for class_idx in range(num_classes):
                # Extract probabilities for this class
                class_probs = mean_softmax[:, class_idx, ...]  # (B, H, W, D)

                # Create binary labels (1 if pixel belongs to this class, 0 otherwise)
                if isinstance(target, (list, tuple)):
                    # Match target resolution to logits resolution
                    target_labels = target[0]  # assume last is highest resolution in DS transform
                else:
                    target_labels = target

                # Squeeze channel dimension if present (B, 1, H, W, D) -> (B, H, W, D)
                if target_labels.dim() == 5 and target_labels.shape[1] == 1:
                    target_labels = target_labels.squeeze(1)

                # Ensure target spatial size matches logits/probs
                if target_labels.shape[2:] != logits.shape[2:]:
                    target_labels = F.interpolate(
                        target_labels.unsqueeze(1).float(),
                        size=logits.shape[2:],
                        mode='nearest'
                    ).squeeze(1)

                binary_labels = (target_labels == class_idx).float()  # (B, H, W, D)

                # Flatten and store - MUST move to CPU before converting to numpy
                all_probs_per_class[class_idx].append(class_probs.cpu().numpy().flatten())
                all_labels_per_class[class_idx].append(binary_labels.cpu().numpy().flatten())

            # Clear GPU memory after each sample
            del inp, logits, mean_softmax, target, batch
            if idx % 5 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}", flush=True)
            # Try to clean up memory on error
            torch.cuda.empty_cache()
            continue

    print("Data collection complete. Concatenating results...")

except Exception as e:
    print(f"\n{'='*70}", flush=True)
    print(f"FATAL ERROR during calibration data collection: {str(e)}", flush=True)
    print(f"{'='*70}", flush=True)
    import traceback
    traceback.print_exc()
    # Force GPU memory cleanup before exit
    torch.cuda.empty_cache()
    sys.exit(1)

finally:
    # Ensure GPU memory is always cleaned up
    print("\nCleaning up GPU memory...", flush=True)
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("GPU memory cleanup complete.", flush=True)

try:
    # Concatenate all samples for each class
    for class_idx in range(num_classes):
        # filter out any empty arrays and ensure equal lengths
        probs_list = [p for p in all_probs_per_class[class_idx] if p.size > 0]
        labels_list = [l for l in all_labels_per_class[class_idx] if l.size > 0]
        all_probs_per_class[class_idx] = np.concatenate(probs_list)
        all_labels_per_class[class_idx] = np.concatenate(labels_list)
        print(f"Class {class_idx}: {len(all_probs_per_class[class_idx])} pixels")

    # ---------------------------------------------------------
    # Option 1: Plot calibration curve for each class separately
    # ---------------------------------------------------------
    print("\nGenerating per-class calibration curves...")
    n_bins = 10
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    ece_per_class = []

    for class_idx in range(num_classes):
        # Skip if class has no positive samples
        if all_labels_per_class[class_idx].sum() == 0:
            print(f"Warning: Class {class_idx} has no positive samples, skipping...")
            axes[class_idx].text(0.5, 0.5, f'Class {class_idx}\nNo positive samples',
                                ha='center', va='center', transform=axes[class_idx].transAxes)
            axes[class_idx].set_xlim([0, 1])
            axes[class_idx].set_ylim([0, 1])
            ece_per_class.append(np.nan)
            continue

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            all_labels_per_class[class_idx],
            all_probs_per_class[class_idx],
            n_bins=n_bins,
            strategy='uniform'
        )

        # Calculate ECE for this class
        ece = np.mean(np.abs(prob_pred - prob_true))
        ece_per_class.append(ece)

        # Plot
        ax = axes[class_idx]
        ax.plot(prob_pred, prob_true, marker='o', color=colors[class_idx],
                label=f'Class {class_idx}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Class {class_idx} (ECE: {ece:.4f})')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f'calibration_curve_per_class_{model_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved per-class calibration curves to calibration_curve_per_class_{model_name}.png")
    plt.close()

    # ---------------------------------------------------------
    # Option 2: Aggregated calibration curve (all classes combined)
    # ---------------------------------------------------------
    print("\nGenerating aggregated calibration curve...")

    # Combine all classes into one array (treating each class as one-vs-rest)
    all_probs_combined = np.concatenate(all_probs_per_class)
    all_labels_combined = np.concatenate(all_labels_per_class)

    # Compute aggregated calibration curve
    prob_true_agg, prob_pred_agg = calibration_curve(
        all_labels_combined,
        all_probs_combined,
        n_bins=n_bins,
        strategy='uniform'
    )

    # Calculate aggregated ECE
    ece_aggregated = np.mean(np.abs(prob_pred_agg - prob_true_agg))

    # Plot aggregated calibration curve
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred_agg, prob_true_agg, marker='o', markersize=8,
             linewidth=2, label='Multi-class Uncertainty')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2,
             label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability (Confidence)', fontsize=12)
    plt.ylabel('Fraction of Positives (Accuracy)', fontsize=12)
    plt.title(f'Reliability Diagram - Aggregated ({num_classes} classes)\nECE: {ece_aggregated:.4f}',
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f'calibration_curve_aggregated_{model_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved aggregated calibration curve to calibration_curve_aggregated_{model_name}.png")
    plt.close()

    # ---------------------------------------------------------
    # Print ECE Summary
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"\nAggregated ECE Score: {ece_aggregated:.4f}")
    print(f"\nPer-class ECE Scores:")
    for class_idx in range(num_classes):
        if not np.isnan(ece_per_class[class_idx]):
            print(f"  Class {class_idx:2d}: {ece_per_class[class_idx]:.4f}")
        else:
            print(f"  Class {class_idx:2d}: N/A (no positive samples)")

    mean_ece_per_class = np.nanmean(ece_per_class)
    print(f"\nMean ECE across classes: {mean_ece_per_class:.4f}")
    print("="*60)

except Exception as e:
    print(f"\n{'='*70}", flush=True)
    print(f"FATAL ERROR during calibration analysis: {str(e)}", flush=True)
    print(f"{'='*70}", flush=True)
    import traceback
    traceback.print_exc()
    # Force GPU memory cleanup before exit
    torch.cuda.empty_cache()
    sys.exit(1)

finally:
    # Ensure GPU memory is always cleaned up
    print("\nCleaning up GPU memory...", flush=True)
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("GPU memory cleanup complete.", flush=True)

# ---------------------------------------------------------
# Temperature Scaling for Calibration
# ---------------------------------------------------------
print("\n" + "="*60)
print("TEMPERATURE SCALING")
print("="*60)

# Split validation data into temp_val (for finding T) and temp_test (for evaluation)
split_idx = len(val_keys) // 2
temp_val_keys = val_keys[:split_idx]
temp_test_keys = val_keys[split_idx:]

print(f"\nSplitting validation set:")
print(f"  Temperature optimization: {len(temp_val_keys)} samples")
print(f"  Temperature evaluation: {len(temp_test_keys)} samples")

# Create datasets
dataset_temp_val = nnUNetDataset(preprocessed_dataset_folder, temp_val_keys,
                                 folder_with_segs_from_previous_stage=None,
                                 num_images_properties_loading_threshold=0)
dataset_temp_test = nnUNetDataset(preprocessed_dataset_folder, temp_test_keys,
                                  folder_with_segs_from_previous_stage=None,
                                  num_images_properties_loading_threshold=0)

# Create data loaders
dl_temp_val = nnUNetDataLoader3D(dataset_temp_val, 1, initial_patch_size, patch_size,
                                 label_manager, oversample_foreground_percent=0.33,
                                 sampling_probabilities=None, pad_sides=None)
dl_temp_test = nnUNetDataLoader3D(dataset_temp_test, 1, initial_patch_size, patch_size,
                                  label_manager, oversample_foreground_percent=0.33,
                                  sampling_probabilities=None, pad_sides=None)

mt_gen_temp_val = LimitedLenWrapper(len(temp_val_keys), data_loader=dl_temp_val,
                                    transform=val_transforms, num_processes=allowed_num_processes,
                                    num_cached=6, seeds=None, pin_memory=True, wait_time=0.02)
mt_gen_temp_test = LimitedLenWrapper(len(temp_test_keys), data_loader=dl_temp_test,
                                     transform=val_transforms, num_processes=allowed_num_processes,
                                     num_cached=6, seeds=None, pin_memory=True, wait_time=0.02)


class TemperatureScaler(nn.Module):
    """
    Temperature scaling model that learns optimal temperature parameter.
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize T=1.5

    def forward(self, logits):
        """
        Apply temperature scaling to logits.
        Args:
            logits: (B, C, H, W, D)
        Returns:
            calibrated probabilities: (B, C, H, W, D)
        """
        return torch.softmax(logits / self.temperature, dim=1)

    def get_temperature(self):
        return self.temperature.item()


def find_optimal_temperature(model, data_loader, num_samples, device):
    """
    Find optimal temperature using NLL loss on validation set.
    Memory-efficient version: stores data on CPU and uses batched optimization.
    """
    print("\nCollecting logits for temperature optimization...")

    all_logits = []
    all_targets = []

    # Use fewer samples to reduce memory usage
    max_samples = min(num_samples, 30)  # Limit to 30 samples for temperature optimization

    model.eval()
    with torch.no_grad():
        for idx in range(max_samples):
            if idx % 10 == 0:
                print(f"  Sample {idx+1}/{max_samples}")

            batch = next(data_loader)
            inp = batch['data'].to(device, non_blocking=True)
            target = batch['target']

            if isinstance(target, (list, tuple)):
                target = target[0].to(device, non_blocking=True)
            else:
                target = target.to(device, non_blocking=True)

            logits = model(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            # Move to CPU immediately to save GPU memory
            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())

            # Clear GPU cache
            del inp, logits, target
            if idx % 5 == 0:
                torch.cuda.empty_cache()

    # Concatenate all batches on CPU
    all_logits = torch.cat(all_logits, dim=0)  # (N, C, H, W, D)
    all_targets = torch.cat(all_targets, dim=0)  # (N, H, W, D)

    print(f"\nOptimizing temperature parameter...")
    print(f"Logits shape: {all_logits.shape}, Targets shape: {all_targets.shape}")

    # Create temperature scaler
    temp_scaler = TemperatureScaler().to(device)

    # Use Adam optimizer instead of LBFGS for better memory efficiency
    optimizer = torch.optim.Adam([temp_scaler.temperature], lr=0.01)

    # Define NLL loss
    criterion = nn.CrossEntropyLoss()

    # Optimize temperature with mini-batches
    best_loss = float('inf')
    best_temp = 1.0
    num_epochs = 50
    batch_size = 2  # Process 2 samples at a time

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        # Process in mini-batches
        for i in range(0, len(all_logits), batch_size):
            batch_logits = all_logits[i:i+batch_size].to(device)
            batch_targets = all_targets[i:i+batch_size].to(device)

            optimizer.zero_grad()

            # Apply temperature scaling
            calibrated_probs = temp_scaler(batch_logits)

            # Reshape for loss calculation
            C = calibrated_probs.shape[1]
            probs_flat = calibrated_probs.permute(0, 2, 3, 4, 1).reshape(-1, C)
            targets_flat = batch_targets.reshape(-1).long()

            # Calculate NLL
            loss = criterion(probs_flat, targets_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Clear GPU memory
            del batch_logits, batch_targets, calibrated_probs, probs_flat, targets_flat

        avg_loss = epoch_loss / num_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_temp = temp_scaler.get_temperature()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}, T = {temp_scaler.get_temperature():.4f}")

        torch.cuda.empty_cache()

    optimal_temp = best_temp
    print(f"Optimal temperature: {optimal_temp:.4f}")

    return optimal_temp


# Find optimal temperature
optimal_temp = find_optimal_temperature(model_dropout, mt_gen_temp_val, len(temp_val_keys), device)

# ---------------------------------------------------------
# Evaluate with Temperature Scaling
# ---------------------------------------------------------
print("\n" + "="*60)
print("EVALUATING WITH TEMPERATURE SCALING")
print("="*60)

all_probs_temp_per_class = [[] for _ in range(num_classes)]
all_labels_temp_per_class = [[] for _ in range(num_classes)]

print(f"\nProcessing {len(temp_test_keys)} test samples with T={optimal_temp:.4f}...")

for idx in range(len(temp_test_keys)):
    if idx % 10 == 0:
        print(f"Processing sample {idx+1}/{len(temp_test_keys)}")

    batch = next(mt_gen_temp_test)
    inp = batch['data'].to(device, non_blocking=True)
    target = batch['target']

    if isinstance(target, (list, tuple)):
        target = [t.to(device, non_blocking=True) for t in target]
    else:
        target = target.to(device, non_blocking=True)

    with torch.no_grad():
        logits = model_dropout(inp)

    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    # Apply temperature scaling
    calibrated_probs = torch.softmax(logits / optimal_temp, dim=1)

    for class_idx in range(num_classes):
        class_probs = calibrated_probs[:, class_idx, ...]

        if isinstance(target, (list, tuple)):
            target_labels = target[0]
        else:
            target_labels = target

        # Squeeze channel dimension if present (B, 1, H, W, D) -> (B, H, W, D)
        if target_labels.dim() == 5 and target_labels.shape[1] == 1:
            target_labels = target_labels.squeeze(1)

        binary_labels = (target_labels == class_idx).float()

        all_probs_temp_per_class[class_idx].append(class_probs.cpu().numpy().flatten())
        all_labels_temp_per_class[class_idx].append(binary_labels.cpu().numpy().flatten())

# Concatenate results
for class_idx in range(num_classes):
    all_probs_temp_per_class[class_idx] = np.concatenate(all_probs_temp_per_class[class_idx])
    all_labels_temp_per_class[class_idx] = np.concatenate(all_labels_temp_per_class[class_idx])

# Calculate ECE with temperature scaling
print("\nCalculating ECE with temperature scaling...")
ece_temp_per_class = []

for class_idx in range(num_classes):
    if all_labels_temp_per_class[class_idx].sum() == 0:
        ece_temp_per_class.append(np.nan)
        continue

    prob_true, prob_pred = calibration_curve(
        all_labels_temp_per_class[class_idx],
        all_probs_temp_per_class[class_idx],
        n_bins=n_bins,
        strategy='uniform'
    )

    ece = np.mean(np.abs(prob_pred - prob_true))
    ece_temp_per_class.append(ece)

# Aggregated results with temperature scaling
all_probs_temp_combined = np.concatenate(all_probs_temp_per_class)
all_labels_temp_combined = np.concatenate(all_labels_temp_per_class)

prob_true_temp_agg, prob_pred_temp_agg = calibration_curve(
    all_labels_temp_combined,
    all_probs_temp_combined,
    n_bins=n_bins,
    strategy='uniform'
)

ece_temp_aggregated = np.mean(np.abs(prob_pred_temp_agg - prob_true_temp_agg))

# ---------------------------------------------------------
# Plot Comparison: Before vs After Temperature Scaling
# ---------------------------------------------------------
print("\nGenerating comparison plots...")

# Comparison plot for aggregated results
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Before temperature scaling
axes[0].plot(prob_pred_agg, prob_true_agg, marker='o', markersize=8,
            linewidth=2, label=f'Before (ECE: {ece_aggregated:.4f})')
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2,
            label='Perfectly Calibrated')
axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
axes[0].set_ylabel('Fraction of Positives', fontsize=12)
axes[0].set_title(f'Before Temperature Scaling\nT=1.0, ECE={ece_aggregated:.4f}', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# After temperature scaling
axes[1].plot(prob_pred_temp_agg, prob_true_temp_agg, marker='o', markersize=8,
            linewidth=2, color='green', label=f'After (ECE: {ece_temp_aggregated:.4f})')
axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2,
            label='Perfectly Calibrated')
axes[1].set_xlabel('Mean Predicted Probability', fontsize=12)
axes[1].set_ylabel('Fraction of Positives', fontsize=12)
axes[1].set_title(f'After Temperature Scaling\nT={optimal_temp:.4f}, ECE={ece_temp_aggregated:.4f}', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'calibration_comparison_{model_name}.png', dpi=150, bbox_inches='tight')
print(f"Saved comparison plot to calibration_comparison_{model_name}.png")
plt.close()

# ---------------------------------------------------------
# Print Final Results
# ---------------------------------------------------------
print("\n" + "="*60)
print("TEMPERATURE SCALING RESULTS")
print("="*60)
print(f"\nOptimal Temperature: {optimal_temp:.4f}")
print(f"\nBefore Temperature Scaling:")
print(f"  Aggregated ECE: {ece_aggregated:.4f}")
print(f"  Mean ECE per class: {mean_ece_per_class:.4f}")

mean_ece_temp_per_class = np.nanmean(ece_temp_per_class)
print(f"\nAfter Temperature Scaling:")
print(f"  Aggregated ECE: {ece_temp_aggregated:.4f}")
print(f"  Mean ECE per class: {mean_ece_temp_per_class:.4f}")

improvement = ((ece_aggregated - ece_temp_aggregated) / ece_aggregated) * 100
print(f"\nImprovement: {improvement:.2f}%")
print("="*60)
