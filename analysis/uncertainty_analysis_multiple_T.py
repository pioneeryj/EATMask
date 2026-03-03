import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import DownsampleSegForDSTransform2
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from typing import Union, Tuple, List
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import pickle
from tqdm import tqdm
import os

# ==================== Model Definition ====================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True

class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_ratio)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.dropout(y)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest', dropout_ratio=0.5):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        self.dropout = nn.Dropout3d(p=dropout_ratio)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        x = self.dropout(x)
        return x

class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True, dropout_ratio=0.5):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.dropout_ratio=dropout_ratio
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                                                stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                                  *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                                    for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                                  *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

# ==================== Data Transform Functions ====================
def get_validation_transforms(deep_supervision_scales: Union[List, Tuple],
                              is_cascaded: bool = False,
                              foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                              regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                              ignore_label: int = None):
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
    from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData
    from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import ConvertSegmentationToRegionsTransform
    from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import DownsampleSegForDSTransform2

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))

    if is_cascaded:
        val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                    if ignore_label is not None else regions,
                                                                    'target', 'target'))

    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                           output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)
    return val_transforms

# ==================== Uncertainty Functions ====================
def monte_carlo(model, inp, T):
    """
    model: STUNET with dropout layer
    inp : input tensor of (B, 1, H, W, D)
    T : number of monte carlo samples
    Returns : List(len=T) of logits tensors, each of list(len=5) for multi-resolution output
    """
    model.train()  # Enable dropout
    logits_list = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(inp)
            logits_list.append(logits)
    return logits_list

def calculate_epistemic_uncertainty(logits_list):
    '''
    logits_list : list of len T, each element list of len 5,
                    each element tensor [b,1,h,w,d]
    output : [b,h,w,d]
    '''
    new_list = [i[0] for i in logits_list]  # 해상도 제일 높은 seg output만 사용
    logits_stack = torch.stack(new_list, dim=0)

    # Apply Sigmoid to convert logits to probabilities
    probs_stack = torch.sigmoid(logits_stack)

    # Mean and variance of probabilities across T samples
    mean_probs = probs_stack.mean(dim=0)
    var_probs = probs_stack.var(dim=0)

    # Squeeze to remove the single channel dimension
    epistemic_uncertainty = var_probs.squeeze(1)
    return epistemic_uncertainty

def calculate_aleatoric_uncertainty(logits_list):
    '''
    logits_list : list of len T, each element list of len 5,
                    each element tensor [b,1,h,w,d]
    output : [b,h,w,d]
    '''
    new_list = [i[0] for i in logits_list]
    logits_stack = torch.stack(new_list, dim=0)
    prob_samples = torch.sigmoid(logits_stack)
    entropy_samples = - (prob_samples * torch.log(prob_samples + 1e-8) +
                         (1 - prob_samples) * torch.log(1 - prob_samples + 1e-8))
    aleatoric_uncertainty = entropy_samples.mean(dim=0).squeeze(1)
    return aleatoric_uncertainty

def compute_uncertainties_streaming(model, inp, T):
    """
    Memory-efficient computation of epistemic (variance of prob) and aleatoric (mean entropy)
    without storing all logits. Returns tensors shaped (B, H, W, D).
    """
    model.train()
    with torch.no_grad():
        sum_probs = None
        sum_probs_sq = None
        sum_entropy = None
        for _ in range(T):
            out = model(inp)
            # use highest resolution output at index 0
            logits = out[0]
            probs = torch.sigmoid(logits)
            entropy = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            if sum_probs is None:
                sum_probs = probs
                sum_probs_sq = probs * probs
                sum_entropy = entropy
            else:
                sum_probs = sum_probs + probs
                sum_probs_sq = sum_probs_sq + probs * probs
                sum_entropy = sum_entropy + entropy
        mean_probs = sum_probs / T
        mean_probs_sq = sum_probs_sq / T
        var_probs = mean_probs_sq - mean_probs * mean_probs
        epistemic = var_probs.squeeze(1)
        aleatoric = (sum_entropy / T).squeeze(1)
        return epistemic, aleatoric

# ==================== Memory-Efficient Uncertainty ====================
def compute_uncertainties_streaming(model, inp, T):
    """
    Compute epistemic (variance of probabilities) and aleatoric (mean entropy)
    uncertainties without storing all T logits in memory.

    Returns:
        epistemic_uncertainty: tensor [b, h, w, d]
        aleatoric_uncertainty: tensor [b, h, w, d]
    """
    model.train()
    sum_p = None
    sum_p2 = None
    sum_entropy = None

    with torch.no_grad():
        for _ in range(T):
            logits = model(inp)
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            probs = torch.sigmoid(logits)

            if sum_p is None:
                sum_p = probs.clone()
                sum_p2 = (probs * probs).clone()
                entropy = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
                sum_entropy = entropy
            else:
                sum_p += probs
                sum_p2 += probs * probs
                entropy = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
                sum_entropy += entropy

            # Explicitly free temps
            del logits, probs, entropy

    mean_p = sum_p / T
    var_p = (sum_p2 / T) - (mean_p * mean_p)
    epistemic_uncertainty = var_p.squeeze(1)
    aleatoric_uncertainty = (sum_entropy / T).squeeze(1)

    # Free accumulators
    del sum_p, sum_p2, sum_entropy, mean_p, var_p
    torch.cuda.empty_cache()

    return epistemic_uncertainty, aleatoric_uncertainty

# ==================== Statistics Functions ====================
def compute_statistics(uncertainty_map):
    """
    uncertainty_map: tensor of shape (B, H, W, D)
    Returns dictionary of statistics
    """
    uncertainty_flat = uncertainty_map.flatten().cpu().numpy()
    return {
        'mean': np.mean(uncertainty_flat),
        'std': np.std(uncertainty_flat),
        'min': np.min(uncertainty_flat),
        'max': np.max(uncertainty_flat),
        'median': np.median(uncertainty_flat),
        'q25': np.percentile(uncertainty_flat, 25),
        'q75': np.percentile(uncertainty_flat, 75)
    }

# ==================== Visualization Functions ====================
def plot_correlation_heatmap(correlation_matrix, T_values, uncertainty_type, save_path):
    """
    Plot correlation heatmap between different T values
    """
    plt.figure(figsize=(10, 8))

    # Create labels
    labels = [f'T={t}' for t in T_values]

    # Plot heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1, center=0, square=True)

    plt.title(f'{uncertainty_type} Uncertainty Correlation Heatmap\n(Across Different T Values)',
              fontsize=14, pad=20)
    plt.xlabel('Monte Carlo Samples (T)', fontsize=12)
    plt.ylabel('Monte Carlo Samples (T)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {save_path}")

def plot_statistics_comparison(stats_dict, T_values, uncertainty_type, save_path):
    """
    Plot statistics comparison across different T values
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{uncertainty_type} Uncertainty Statistics Across T Values', fontsize=16)

    metrics = ['mean', 'std', 'median', 'min', 'max', 'q75']

    for idx, (ax, metric) in enumerate(zip(axes.flatten(), metrics)):
        values = [stats_dict[t][metric] for t in T_values]
        ax.plot(T_values, values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('T (Monte Carlo Samples)', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'{metric.capitalize()} vs T', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics plot to: {save_path}")

def save_maps_grid(maps_per_t, T_values, uncertainty_type, save_path, base_image=None, alpha: float = 0.6):
    """
    Save a 1x6 grid figure for uncertainty maps across T values.
    maps_per_t: Dict[T, torch.Tensor] with shape (1, H, W, D)
    """
    import matplotlib.pyplot as plt
    n_cols = len(T_values)
    fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 3))
    # choose central slice in D
    for idx, t in enumerate(T_values):
        ax = axes[idx] if n_cols > 1 else axes
        m = maps_per_t[t]
        if m.ndim == 4:
            # (B,H,W,D)
            m_np = m[0].cpu().numpy()
            d_mid = m_np.shape[-1]//2
            img = m_np[..., d_mid]
        else:
            img = m.squeeze().cpu().numpy()
        # Base image overlay if provided (expect shape (B, C=1, H, W, D))
        if base_image is not None:
            bi = base_image
            if bi.ndim == 5:
                bi_np = bi[0, 0].cpu().numpy()  # (H,W,D)
                d_mid_b = bi_np.shape[-1]//2
                base_slice = bi_np[..., d_mid_b]
            elif bi.ndim == 3:
                base_slice = bi
            else:
                base_slice = None
            if base_slice is not None:
                # Normalize base to [0,1]
                bmin, bmax = np.min(base_slice), np.max(base_slice)
                if bmax > bmin:
                    base_norm = (base_slice - bmin) / (bmax - bmin)
                else:
                    base_norm = base_slice
                ax.imshow(base_norm, cmap='gray')
        im = ax.imshow(img, cmap='jet', alpha=alpha)
        ax.set_title(f'T={t}')
        ax.axis('off')
    fig.suptitle(f'{uncertainty_type} map across T')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# ==================== Main Analysis Function ====================
def run_uncertainty_analysis(model, dataloader, device, T_values, num_samples, output_dir):
    """
    Run uncertainty analysis for multiple T values

    Args:
        model: STUNet model
        dataloader: Data loader
        device: torch device
        T_values: List of T values for Monte Carlo sampling
        num_samples: Number of samples to analyze
        output_dir: Directory to save results
    """
    maybe_mkdir_p(output_dir)

    # Storage for results
    epistemic_results = {t: [] for t in T_values}
    aleatoric_results = {t: [] for t in T_values}
    # Timing accumulators per T
    time_total_per_t = {t: 0.0 for t in T_values}
    time_count_per_t = {t: 0 for t in T_values}

    print(f"\n{'='*60}")
    print(f"Starting Uncertainty Analysis")
    print(f"Number of samples: {num_samples}")
    print(f"T values: {T_values}")
    print(f"{'='*60}\n")

    # Process samples
    sample_count = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if sample_count >= num_samples:
            break

        inp = batch['data'].to(device, non_blocking=True)
        batch_size = inp.shape[0]

        # Process each sample in the batch
        for sample_idx in range(batch_size):
            if sample_count >= num_samples:
                break

            inp_single = inp[sample_idx:sample_idx+1]  # Keep batch dimension

            # Run Monte Carlo for each T value and collect per-sample maps
            per_sample_epistemic = {}
            per_sample_aleatoric = {}
            for T in T_values:
                # Memory-efficient uncertainties
                import time
                start_t = time.perf_counter()
                epistemic_map, aleatoric_map = compute_uncertainties_streaming(model, inp_single, T)
                elapsed = time.perf_counter() - start_t
                time_total_per_t[T] += elapsed
                time_count_per_t[T] += 1
                # Store results for aggregate stats
                epistemic_results[T].append(epistemic_map.cpu())
                aleatoric_results[T].append(aleatoric_map.cpu())
                # Collect per-sample for plotting
                per_sample_epistemic[T] = epistemic_map.detach().cpu()
                per_sample_aleatoric[T] = aleatoric_map.detach().cpu()
                torch.cuda.empty_cache()

            # Save 1x6 PNGs for this sample
            sample_dir = join(output_dir, f"sample_{sample_count:03d}")
            maybe_mkdir_p(sample_dir)
            save_maps_grid(per_sample_epistemic, T_values, 'Epistemic', join(sample_dir, 'epistemic_1x6.png'), base_image=inp_single)
            save_maps_grid(per_sample_aleatoric, T_values, 'Aleatoric', join(sample_dir, 'aleatoric_1x6.png'), base_image=inp_single)

            sample_count += 1

            if (sample_count) % 10 == 0:
                print(f"Processed {sample_count}/{num_samples} samples")

    print(f"\nCompleted processing {sample_count} samples")

    # Compute statistics for each T
    print("\n" + "="*60)
    print("Computing Statistics")
    print("="*60 + "\n")

    epistemic_stats = {}
    aleatoric_stats = {}

    for T in T_values:
        # Concatenate all samples for this T
        epistemic_all = torch.cat(epistemic_results[T], dim=0)
        aleatoric_all = torch.cat(aleatoric_results[T], dim=0)

        # Compute statistics
        epistemic_stats[T] = compute_statistics(epistemic_all)
        aleatoric_stats[T] = compute_statistics(aleatoric_all)

        print(f"\nT = {T}:")
        print(f"  Epistemic - Mean: {epistemic_stats[T]['mean']:.6f}, Std: {epistemic_stats[T]['std']:.6f}")
        print(f"  Aleatoric - Mean: {aleatoric_stats[T]['mean']:.6f}, Std: {aleatoric_stats[T]['std']:.6f}")

    # Save statistics to CSV
    epistemic_df = pd.DataFrame(epistemic_stats).T
    aleatoric_df = pd.DataFrame(aleatoric_stats).T

    epistemic_df.to_csv(join(output_dir, 'epistemic_statistics.csv'))
    aleatoric_df.to_csv(join(output_dir, 'aleatoric_statistics.csv'))
    print(f"\nSaved statistics to CSV files in {output_dir}")

    # Save timing per T to txt
    timing_lines = ["Per-T computation time (seconds)\n"]
    for T in T_values:
        total = time_total_per_t[T]
        count = time_count_per_t[T]
        avg = total / count if count > 0 else 0.0
        timing_lines.append(f"T={T}: total={total:.4f}s, count={count}, avg={avg:.4f}s\n")
    with open(join(output_dir, 'per_T_timing.txt'), 'w') as f:
        f.writelines(timing_lines)
    print(f"Saved timing log to: {join(output_dir, 'per_T_timing.txt')}")

    # Compute correlation matrices
    print("\n" + "="*60)
    print("Computing Correlation Matrices")
    print("="*60 + "\n")

    n_t = len(T_values)
    epistemic_corr = np.zeros((n_t, n_t))
    aleatoric_corr = np.zeros((n_t, n_t))

    for i, t1 in enumerate(T_values):
        for j, t2 in enumerate(T_values):
            # Flatten all samples for correlation computation
            epistemic_flat1 = torch.cat(epistemic_results[t1], dim=0).flatten().numpy()
            epistemic_flat2 = torch.cat(epistemic_results[t2], dim=0).flatten().numpy()
            aleatoric_flat1 = torch.cat(aleatoric_results[t1], dim=0).flatten().numpy()
            aleatoric_flat2 = torch.cat(aleatoric_results[t2], dim=0).flatten().numpy()

            # Compute Pearson correlation
            epistemic_corr[i, j], _ = pearsonr(epistemic_flat1, epistemic_flat2)
            aleatoric_corr[i, j], _ = pearsonr(aleatoric_flat1, aleatoric_flat2)

    print("Epistemic Correlation Matrix:")
    print(epistemic_corr)
    print("\nAleatoric Correlation Matrix:")
    print(aleatoric_corr)

    # Save correlation matrices
    np.save(join(output_dir, 'epistemic_correlation_matrix.npy'), epistemic_corr)
    np.save(join(output_dir, 'aleatoric_correlation_matrix.npy'), aleatoric_corr)

    # Plot correlation heatmaps
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60 + "\n")

    plot_correlation_heatmap(epistemic_corr, T_values, 'Epistemic',
                            join(output_dir, 'epistemic_correlation_heatmap.png'))
    plot_correlation_heatmap(aleatoric_corr, T_values, 'Aleatoric',
                            join(output_dir, 'aleatoric_correlation_heatmap.png'))

    # Plot statistics comparison
    plot_statistics_comparison(epistemic_stats, T_values, 'Epistemic',
                              join(output_dir, 'epistemic_statistics_plot.png'))
    plot_statistics_comparison(aleatoric_stats, T_values, 'Aleatoric',
                              join(output_dir, 'aleatoric_statistics_plot.png'))

    # Save raw results
    print("\nSaving raw results...")
    with open(join(output_dir, 'epistemic_results.pkl'), 'wb') as f:
        pickle.dump(epistemic_results, f)
    with open(join(output_dir, 'aleatoric_results.pkl'), 'wb') as f:
        pickle.dump(aleatoric_results, f)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")

    return epistemic_stats, aleatoric_stats, epistemic_corr, aleatoric_corr

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]]
    conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

    # Data configuration
    preprocessed_dataset_folder = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans_3d_fullres'
    splits_file = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/splits_final.json'
    dataset_json_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/dataset.json'
    plans_path = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans.json'
    pretrained_model_path = "/mnt/HDD/yoonji/medmim/pretrained_model/large_ep4k.model"

    # Analysis configuration
    T_VALUES = [5, 10, 15, 20, 30, 50]
    NUM_SAMPLES = 5
    BATCH_SIZE = 1  # Process one sample at a time for consistency
    OUTPUT_DIR = '/mnt/HDD/yoonji/medmim/uncertainty_analysis_results'

    # Load data configuration
    print("Loading data configuration...")
    splits = load_json(splits_file)
    fold = 0
    all_keys = splits[fold]['train']
    tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

    # Use validation set for uncertainty analysis
    dataset_val = nnUNetDataset(preprocessed_dataset_folder, val_keys,
                                folder_with_segs_from_previous_stage=None,
                                num_images_properties_loading_threshold=0)

    dataset_json = load_json(dataset_json_path)
    plans = load_json(plans_path)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    label_manager = plans_manager.get_label_manager(dataset_json)

    # Data loader
    patch_size = [112, 112, 128]
    initial_patch_size = patch_size

    dl_val = nnUNetDataLoader3D(dataset_val, BATCH_SIZE,
                                initial_patch_size,
                                configuration_manager.patch_size,
                                label_manager,
                                oversample_foreground_percent=0.33,
                                sampling_probabilities=None, pad_sides=None)

    # Setup transforms
    deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
        pool_op_kernel_sizes), axis=0))[:-1]

    val_transforms = get_validation_transforms(
        deep_supervision_scales,
        is_cascaded=False,
        foreground_labels=label_manager.foreground_labels,
        regions=label_manager.foreground_regions if label_manager.has_regions else None,
        ignore_label=label_manager.ignore_label)

    allowed_num_processes = get_allowed_n_proc_DA()
    iters_val = min(len(dataset_val), NUM_SAMPLES)

    mt_gen_val = LimitedLenWrapper(iters_val, data_loader=dl_val, transform=val_transforms,
                                   num_processes=allowed_num_processes, num_cached=3, seeds=None,
                                   pin_memory=True, wait_time=0.02)

    # Initialize model
    print("\nInitializing model...")
    model = STUNet(
        input_channels=1,
        num_classes=1,
        depth=[1, 1, 1, 1, 1, 1],
        dims=[32, 64, 128, 256, 512, 512],
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        enable_deep_supervision=True,
        dropout_ratio=0.5
    )

    # Load pretrained weights
    print(f"Loading pretrained weights from: {pretrained_model_path}")
    # Load checkpoint on CPU to avoid GPU memory spike
    pretrained_weights = torch.load(pretrained_model_path, map_location='cpu')
    model.load_state_dict(pretrained_weights, strict=False)
    del pretrained_weights
    model = model.to(device)
    torch.cuda.empty_cache()

    print("Model loaded successfully!")

    # Run uncertainty analysis
    epistemic_stats, aleatoric_stats, epistemic_corr, aleatoric_corr = run_uncertainty_analysis(
        model=model,
        dataloader=mt_gen_val,
        device=device,
        T_values=T_VALUES,
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIR
    )

    print("\nAll done!")
