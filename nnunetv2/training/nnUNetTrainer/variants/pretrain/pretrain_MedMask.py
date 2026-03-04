# region Description of the region
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import device_count
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
from nnunetv2.training.lr_scheduler.LinearWarmupCosine import LinearWarmupCosineAnnealingLR
from STUNet_head import STUNet
from STUNet_dropout import STUNet_dropout

from encoder3D import SparseEncoder
from decoder3D import LightDecoder
from AnatoMask import SparK, monte_carlo, calculate_softmax, calculate_conditional_entropy,calculate_epistemic_uncertainty, calculate_aleatoric_uncertainty

from torch.cuda.amp import GradScaler, autocast
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
from tqdm import tqdm

import math
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from utils.lr_control import lr_wd_annealing, get_param_groups

import torch.nn as nn
import torch.nn.functional as F
import argparse

# ===== Feature analysis helpers =====
import torch.nn.functional as F

def get_foreground_mask_from_input(input_img, feature_map):
    """
    Create a foreground mask from input image and resize to feature_map spatial size.
    input_img: (B, 1, D, H, W)
    feature_map: (B, C, d, h, w)
    Returns: (B, d, h, w) boolean mask
    """
    with torch.no_grad():
        input_mask = (torch.abs(input_img) > 1e-5).float()
        target_size = feature_map.shape[2:]
        mask_resized = F.interpolate(input_mask, size=target_size, mode='nearest')
        mask_final = (mask_resized.squeeze(1) > 0.5)
    return mask_final
def compute_norm_distribution(feat_list, input_img=None):
    norms = []
    with torch.no_grad():
        for f in feat_list:
            # compute L2 norm across channel dim, flatten spatial
            n = torch.linalg.vector_norm(f, ord=2, dim=1)  # (B, H, W, D)
            if input_img is not None:
                # build mask per feature map size
                m = get_foreground_mask_from_input(input_img, f)
                valid = n[m]
                norms.append(valid.detach().cpu().numpy().ravel())
            else:
                norms.append(n.detach().cpu().numpy().ravel())
    if len(norms) == 0:
        return np.array([])
    return np.concatenate(norms)

def compute_feature_entropy(feat_list, temperature=1.0, input_img=None):
    ents = []
    with torch.no_grad():
        for f in feat_list:
            # softmax over channels to interpret as distribution
            p = torch.softmax(f / temperature, dim=1).clamp_min(1e-8)
            ent = -(p * torch.log(p)).sum(dim=1)  # (B, H, W, D)
            if input_img is not None:
                m = get_foreground_mask_from_input(input_img, f)
                valid = ent[m]
                ents.append(valid.detach().cpu().numpy().ravel())
            else:
                ents.append(ent.detach().cpu().numpy().ravel())
    if len(ents) == 0:
        return np.array([])
    return np.concatenate(ents)

def save_hist(data_before, data_after, title, out_path, bins=50):
    try:
        plt.figure(figsize=(8,5))
        if data_before.size > 0:
            plt.hist(data_before, bins=bins, alpha=0.5, label='before')
        if data_after.size > 0:
            plt.hist(data_after, bins=bins, alpha=0.5, label='after')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        maybe_mkdir_p(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=120)
        plt.close()
    except Exception as e:
        print('Failed to save hist:', title, e)

def compute_singular_values_inputmask(feat_list, input_img, max_samples=2048):
    """
    Compute singular values on foreground-only features using input image mask.
    - feat_list: list of feature maps, each (B, C, H, W, D)
    - input_img: (B, 1, D, H, W) input tensor
    - Returns: list of numpy arrays of normalized singular values per feature map
    """
    sv_list = []
    with torch.no_grad():
        for f in feat_list:
            # build per-level foreground mask from input image resized to f spatial size
            m = get_foreground_mask_from_input(input_img, f)  # (B, H, W, D) bool
            # move channels last and flatten spatial + batch to N
            f_perm = f.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
            # apply mask per sample then concatenate
            B = f.shape[0]
            chunks = []
            for b in range(B):
                fg_idx = m[b]
                if fg_idx.sum() == 0:
                    continue
                fb = f_perm[b][fg_idx]  # (N_fg_b, C)
                chunks.append(fb)
            if len(chunks) == 0:
                continue
            f_fg = torch.cat(chunks, dim=0)  # (N_fg, C)
            # sampling for speed
            if f_fg.shape[0] > max_samples:
                idx = torch.randperm(f_fg.shape[0], device=f_fg.device)[:max_samples]
                f_fg = f_fg[idx]
            # center
            f_fg = f_fg - f_fg.mean(dim=0, keepdim=True)
            # SVD
            try:
                _, S, _ = torch.linalg.svd(f_fg, full_matrices=False)
                if S[0] > 0:
                    S = S / S[0]
                sv_list.append(S.detach().cpu().numpy())
            except Exception as e:
                print(f"SVD error: {e}")
                continue
    return sv_list

def plot_singular_values(sv_list, out_path):
    """Plot multiple singular value curves and save to out_path."""
    try:
        plt.figure(figsize=(10,6))
        for i, S in enumerate(sv_list):
            plt.plot(S, label=f'feat_{i}', alpha=0.8)
        plt.title('Normalized Singular Values (foreground-only)')
        plt.xlabel('index')
        plt.ylabel('value (normalized)')
        if len(sv_list) <= 10:
            plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        maybe_mkdir_p(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=120)
        plt.close()
    except Exception as e:
        print('Failed to save SVD plot:', e)

def get_args():
    parser = argparse.ArgumentParser(description='masking_ratio조정')
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--intensity', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="tmp")
    parser.add_argument('--local-rank', type=int, default=-1)
    return parser.parse_args()

args = get_args()

# Initialize DDP
is_ddp = False
if dist.is_available():
    try:
        # Only initialize if env vars present (env://)
        if os.environ.get('RANK') is not None and os.environ.get('WORLD_SIZE') is not None:
            dist.init_process_group(backend='nccl')
            is_ddp = True
    except Exception as e:
        print('DDP init failed, falling back to single-GPU:', e)

if is_ddp:
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(type='cuda', index=local_rank)
    print(f"I am local rank {local_rank}. {device_count()} GPUs are available. The world size is "
          f"{world_size}. Setting device to {device}")
else:
    local_rank = 0
    world_size = 1
    device = torch.device("cuda:0")

base_folder = "/nas_homes/yoonji/medmask/nnUNet_results/pretraining"
output_folder = join(base_folder, args.model_name, str(args.mask_ratio))
# Training transforms, data augmentation pipelines
def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                            rotation_for_DA: dict,
                            deep_supervision_scales: Union[List, Tuple],
                            mirror_axes: Tuple[int, ...],
                            do_dummy_2d_data_aug: bool,
                            order_resampling_data: int = 3,
                            order_resampling_seg: int = 1,
                            border_val_seg: int = -1,
                            use_mask_for_norm: List[bool] = None,
                            is_cascaded: bool = False,
                            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                            ignore_label: int = None) -> AbstractTransform:
    tr_transforms = []
    if do_dummy_2d_data_aug:
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    # First augmentation transform, dont change
    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    if do_dummy_2d_data_aug:
        tr_transforms.append(Convert2DTo3DTransform())

    # CHANGE HERE!!!!!!

    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    # tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
    #                                            p_per_channel=0.5))
    # tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    # tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    # tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
    #                                                     p_per_channel=0.5,
    #                                                     order_downsample=0, order_upsample=3, p_per_sample=0.25,
    #                                                     ignore_axes=ignore_axes))
    # tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    # tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    if use_mask_for_norm is not None and any(use_mask_for_norm):
        tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                           mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if is_cascaded:
        assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
        tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
        tr_transforms.append(ApplyRandomBinaryOperatorTransform(
            channel_idx=list(range(-len(foreground_labels), 0)),
            p_per_sample=0.4,
            key="data",
            strel_size=(1, 8),
            p_per_label=1))
        tr_transforms.append(
            RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                key="data",
                p_per_sample=0.2,
                fill_with_other_class_p=0,
                dont_do_if_covers_more_than_x_percent=0.15))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        # the ignore label must also be converted
        tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                   if ignore_label is not None else regions,
                                                                   'target', 'target'))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                          output_key='target'))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

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

# end region

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


# input size
input_size = (112, 112, 128)

enc = SparseEncoder(head, input_size=input_size, sbn=is_ddp).to(device)
dec = LightDecoder(enc.downsample_ratio, sbn=False, width=512, out_channel=1).to(device)

teacher_feature_dims = [512, 256, 128, 64, 32]

model_without_ddp = SparK(
    sparse_encoder=enc, dense_decoder=dec, mask_ratio=args.mask_ratio,
    densify_norm='in', teacher_feature_dims=teacher_feature_dims
).to(device)

# model_without_ddp = torch.compile(model_without_ddp)

model_ema = ModelEma(model_without_ddp, decay=0.999, device=device, resume='')

if is_ddp:
    model = DDP(model_without_ddp, device_ids=[local_rank],
                find_unused_parameters=True, broadcast_buffers=False)
else:
    model = model_without_ddp


# # Change this every time...
fold = 0
epoch = 501
global_batch_size = 4
global_oversample_percents = 0.33
opt = 'adamw'
ada = 0.999
lr = 1e-4
weight_decay = 1e-5
clip = 12
wd = 0.04
wde = 0.2
wp_ep = 8
warmup = 20
AMP = False
guide = True
alpha = 0.9

# DDP batch size distribution
batch_sizes = []
oversample_percents = []

assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of GPUs... Duh.'

batch_size_per_GPU = int(np.ceil(global_batch_size / world_size))

for rank in range(world_size):
    if (rank + 1) * batch_size_per_GPU > global_batch_size:
        batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
    else:
        batch_size = batch_size_per_GPU

    batch_sizes.append(batch_size)

    sample_id_low = 0 if len(batch_sizes) == 0 else int(np.sum(batch_sizes[:-1]))
    sample_id_high = int(np.sum(batch_sizes))

    if sample_id_high / global_batch_size < (1 - global_oversample_percents):
        oversample_percents.append(0.0)
    elif sample_id_low / global_batch_size > (1 - global_oversample_percents):
        oversample_percents.append(1.0)
    else:
        percent_covered_by_this_rank = sample_id_high / global_batch_size - sample_id_low / global_batch_size
        oversample_percent_here = 1 - (((1 - global_oversample_percents) -
                                        sample_id_low / global_batch_size) / percent_covered_by_this_rank)
        oversample_percents.append(oversample_percent_here)

if local_rank == 0:
    print("worker", local_rank, "oversample", oversample_percents[local_rank])
    print("worker", local_rank, "batch_size", batch_sizes[local_rank])

batch_size = batch_sizes[local_rank]
oversample_foreground_percent = oversample_percents[local_rank]


timestamp = datetime.now()
if local_rank == 0:
    maybe_mkdir_p(output_folder)

log_file = join(output_folder, 'training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt'%
                     (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                      timestamp.second))


# lut 에서 사용된 feature distance (broader contextualization loss)
def loss_ftn(output1, output2):
    output1 = F.normalize(output1, dim=-1, p=2)
    output2 = F.normalize(output2, dim=-1, p=2)
    return 2 - 2 * (output1 * output2).sum(dim=-1)

# cosine similarity loss
def loss_cosine(emb_list1, emb_list2):
    '''
    item shape of the list is (b,1,f,f)
    len of the list is five
    '''
    cosine_loss_total = 0
    for emb1, emb2 in zip(emb_list1, emb_list2):
        b = emb1.shape[0]
        emb1_flat = emb1.view(b,-1)
        emb2_flat = emb2.view(b,-1)

        cosine_sim = F.cosine_similarity(emb1_flat, emb2_flat, dim=1)
        cosine_loss = 1 - cosine_sim.mean()
        cosine_loss_total += cosine_loss
    cosine_loss_total/=len(emb_list1)
    return cosine_loss_total

def print_to_log_file(*args, also_print_to_console=True, add_timestamp=True):
    if local_rank != 0:
        return

    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = ("%s:" % dt_object, *args)

    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file, 'a+') as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)

### Your preprocessed dataset folder
preprocessed_dataset_folder = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans_3d_fullres'
### Your nnUNet splits json
splits_file = '/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/splits_final.json'
splits = load_json(splits_file)

all_keys = splits[fold]['train']
tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

dataset_tr = nnUNetDataset(preprocessed_dataset_folder, tr_keys,
                           folder_with_segs_from_previous_stage=None,
                           num_images_properties_loading_threshold=0)
dataset_val = nnUNetDataset(preprocessed_dataset_folder, val_keys,
                            folder_with_segs_from_previous_stage=None,
                            num_images_properties_loading_threshold=0)
### Your nnUNet dataset json
dataset_json =load_json('/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/dataset.json')
### Your nnUNet plans json
plans = load_json('/nas_homes/yoonji/medmask/nnUNet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans.json')
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

dl_tr = nnUNetDataLoader3D(dataset_tr, batch_size,
                           initial_patch_size,
                           patch_size,
                           label_manager,
                           oversample_foreground_percent=oversample_foreground_percent,
                           sampling_probabilities=None, pad_sides=None)

iters_train = len(dataset_tr) // batch_size

deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            pool_op_kernel_sizes), axis=0))[:-1]
mirror_axes = (0, 1, 2)

tr_transforms = get_training_transforms(
    patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, False,
    order_resampling_data=3, order_resampling_seg=1,
    use_mask_for_norm=configuration_manager.use_mask_for_norm,
    is_cascaded=False, foreground_labels=label_manager.foreground_labels,
    regions=label_manager.foreground_regions if label_manager.has_regions else None,
    ignore_label=label_manager.ignore_label)

val_transforms = get_validation_transforms(
    deep_supervision_scales,
    is_cascaded=False,
    foreground_labels=label_manager.foreground_labels,
    regions=label_manager.foreground_regions if
    label_manager.has_regions else None,
    ignore_label=label_manager.ignore_label)

allowed_num_processes = 2

mt_gen_train = LimitedLenWrapper(iters_train, data_loader=dl_tr, transform=tr_transforms,
                                 num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                 pin_memory= True, wait_time=0.02)


# # build optimizer and lr_scheduler
param_groups = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})
# unified access for DDP/non-DDP
base_model = model.module if (is_ddp and hasattr(model, 'module')) else model
projection_params = list(base_model.projection_head.parameters())

for group in param_groups:
    group['params'] = [p for p in group['params'] if not any(id(p) == id(q) for q in projection_params)]

param_groups.append({
    'params': projection_params,
    'lr': lr,  # 만약 checkpoint의 lr 값을 그대로 사용하고 싶다면 9.572063115079066e-05로 대체 가능합니다.
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': weight_decay,  # checkpoint에서는 1e-05로 저장됨
    'amsgrad': False,
    'foreach': None,
    'maximize': False,
    'capturable': False,
    'differentiable': False,
    'fused': None,
})
opt_clz = {
    'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
    'adamw': partial(torch.optim.AdamW, betas=(0.9, ada)),
    #     'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, ada), max_grad_norm=5.0),
}[opt]

optimizer = opt_clz(params=param_groups, lr=lr, weight_decay=weight_decay)
# len(optimizer.state_dict()['param_groups'][0].keys() = 13

# print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup, epoch, 1e-6)
# optimizer.load_state_dict(checkpoint['optimizer_state'])


################################## load 하는 부분 ############################################################################
# 어디서 불러올지
file_name = f"{args.model_name}_head_latest.pt"
checkpoint_path = join(output_folder, file_name)
# checkpoint_path = join(output_folder, 'no_checkpoint')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['network_weights']
    optimizer_state = checkpoint['optimizer_state']

    if is_ddp:
        # For DDP, add 'module.' prefix if not present
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("module."):
                new_state_dict["module." + key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    else:
        # For non-DDP, remove 'module.' prefix if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)

    # optimizer (adam)
    if 'initial_lr' in optimizer_state:
        current_lr = optimizer.defaults.get('lr', None)
        if current_lr is not None:
            optimizer_state['initial_lr'] = current_lr

    optimizer.load_state_dict(optimizer_state)
    start_epoch = checkpoint['current_epoch'] + 1
    if local_rank == 0:
        print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0  
#####################################################################################################################

it = 0
epoch_loss = []
epoch_ema_loss = []
optimizer.zero_grad()

# scaler = GradScaler()
logger = nnUNetLogger()

# STUNet model with dropout for uncertainty estimation
model_dropout = STUNet_dropout(
    input_channels=1,
    num_classes=1,
    depth=[1, 1, 1, 1, 1, 1],
    dims=[32, 64, 128, 256, 512, 512],
    pool_op_kernel_sizes=pool_op_kernel_sizes,
    conv_kernel_sizes=conv_kernel_sizes,
    enable_deep_supervision=True,
    dropout_ratio=0.5
)

pretrained_model = "/home/yoonji/AnatoMask/nnunetv2/training/nnUNetTrainer/variants/pretrain/pretrained_model/large_ep4k.model"  # pretrained weights 경로
pretrained_weights = torch.load(pretrained_model, map_location = device, weights_only=False)
model_dropout.load_state_dict(pretrained_weights, strict=False)
model_dropout = model_dropout.to(device)

# make argparser
w_mae = 1
w_cos = 1
w_p = 1

# ===== Analysis accumulators (epoch-wise stats) =====
analysis_stats = {
    'epoch': [],
    'norm_before_mean': [], 'norm_before_std': [],
    'norm_after_mean': [], 'norm_after_std': [],
    'ent_before_mean': [], 'ent_before_std': [],
    'ent_after_mean': [], 'ent_after_std': [],
}


for epoch_idx in range(start_epoch,epoch):
    model.train()
    per_loss = 0.0
    per_p_loss = 0.0
    print_to_log_file('')
    print_to_log_file(f'Epoch {epoch_idx}')
    print_to_log_file()

    print_to_log_file(
        f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}")
    logger.log('epoch_start_timestamps', time.time(), epoch_idx)
    # add this
    if epoch_idx < epoch//4:
        model_ema.decay = 0.999 + epoch_idx / (epoch//4) * (0.9999 - 0.999)
    else:
        model_ema.decay = 0.9999

    last_student_feats = None
    for idx in tqdm(range(iters_train), desc=f"Epoch {epoch_idx} training", leave=False):

        inp = next(mt_gen_train)
        inp = inp['data']
        inp = inp.to(device, non_blocking=True) # (4, 1, 128, 128, 128)


        logit_list = monte_carlo(model=model_dropout, inp=inp, T=5)
        epistemic_map = calculate_epistemic_uncertainty(logit_list).unsqueeze(1)
        aleatoric_map = calculate_aleatoric_uncertainty(logit_list)
# shape: 4,1,7,7,8
        
        mask_union_uncertainty = base_model.mask_uncertainty(batch_size, device, epistemic_map, aleatoric_map, args.mask_ratio) 
        mask_epistemic_intensity = base_model.mask_intensity(batch_size, device, uncertainty_map=epistemic_map, intensity=args.intensity, masking_ratio=args.mask_ratio)
        mask_aleatoric_intensity = base_model.mask_intensity(batch_size, device, uncertainty_map=aleatoric_map, intensity=args.intensity, masking_ratio=args.mask_ratio)

        ### add for ablation(single uncertainty)
        # mask_single_uncertainty = model.module.mask_single_uncertainty(batch_size, device, aleatoric_map, args.mask_ratio) 

        with torch.no_grad():
            #rec_t, inp_t = model_ema.ema(inp, active_b1ff=mask_epistemic_intensity)
            #rec_t_al, inp_t_al = model_ema.ema(inp, active_b1ff=mask_aleatoric_intensity)

            teacher_ep_output = model_ema.ema.forward_encoder(inp, active_b1ff=mask_epistemic_intensity)
            teacher_al_output = model_ema.ema.forward_encoder(inp, active_b1ff=mask_aleatoric_intensity)
            # each output with list len 5 (ex, [0]의 경우 [4, 512,7,7,8],  
            # [3]의 경우 [4,64,56,56,64], [4]의 경우 [4,32,112,112,128])
        
        # Apply projection head to each feature level separately for each uncertainty type
        proj_ep_output = []
        proj_al_output = []
        for lvl in range(len(teacher_ep_output)):
            proj_ep = base_model.projection_head[lvl](teacher_ep_output[lvl])
            proj_al = base_model.projection_head[lvl](teacher_al_output[lvl])
            proj_ep_output.append(proj_ep)
            proj_al_output.append(proj_al)


        rec_s, inp_s = model(inp, active_b1ff=mask_union_uncertainty)
        #rec_single, inp_single = model(inp, active_b1ff=mask_single_uncertainty)



        student_output = base_model.forward_encoder(inp,active_b1ff=mask_union_uncertainty)

        # student reconstruction loss
        loss_p,_ = base_model.forward_loss(inp_s, rec_s, mask_union_uncertainty)
        # feature level loss - compute cosine loss for both uncertainty types
        print(proj_ep_output[0].shape)
        print(proj_al_output[0].shape)
        print(student_output[0].shape)
        cos_loss = loss_cosine(proj_ep_output+proj_al_output, student_output)

        

        ## change to single for ablataion
        loss = w_cos * cos_loss + w_p*loss_p

        # ===== Feature analysis: before alignment (current weights) =====
        # teacher projected as alignment target, student current features
        teacher_feats = proj_ep_output + proj_al_output
        student_feats = student_output
        last_student_feats = student_feats
        # compute per-level foreground masks implicitly inside helpers
        norms_before = compute_norm_distribution(student_feats, input_img=inp)
        ents_before = compute_feature_entropy(student_feats, input_img=inp)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
        optimizer.step()
        model_ema.update(model)

        # Save quick histograms per epoch (overwrite within epoch) - before only
        if local_rank == 0:
            analysis_dir = join(output_folder, 'analysis')
            save_hist(norms_before, np.array([]),
                      title=f'Feature L2 Norms (before only, epoch {epoch_idx})',
                      out_path=join(analysis_dir, f'feature_norms_epoch_{epoch_idx}.png'))
            save_hist(ents_before, np.array([]),
                      title=f'Feature Entropy (before only, epoch {epoch_idx})',
                      out_path=join(analysis_dir, f'feature_entropy_epoch_{epoch_idx}.png'))

        # Accumulate epoch-wise means/stds
        nb_mean = float(np.mean(norms_before)) if norms_before.size > 0 else float('nan')
        nb_std  = float(np.std(norms_before))  if norms_before.size > 0 else float('nan')
        eb_mean = float(np.mean(ents_before))  if ents_before.size > 0 else float('nan')
        eb_std  = float(np.std(ents_before))   if ents_before.size > 0 else float('nan')

        # Log the epoch index (not the inner iteration index)
        analysis_stats['epoch'].append(epoch_idx)
        analysis_stats['norm_before_mean'].append(nb_mean)
        analysis_stats['norm_before_std'].append(nb_std)
        analysis_stats['ent_before_mean'].append(eb_mean)
        analysis_stats['ent_before_std'].append(eb_std)

        # Write CSV progressively and update epoch-wise plot
        if local_rank == 0:
            analysis_dir = join(output_folder, 'analysis')
            maybe_mkdir_p(analysis_dir)
            csv_path = join(analysis_dir, 'feature_stats.csv')
            # write/overwrite full CSV each epoch for simplicity
            try:
                import csv
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch','norm_before_mean','norm_before_std',
                                     'ent_before_mean','ent_before_std'])
                    for k in range(len(analysis_stats['epoch'])):
                        writer.writerow([
                            analysis_stats['epoch'][k],
                            analysis_stats['norm_before_mean'][k], analysis_stats['norm_before_std'][k],
                            analysis_stats['ent_before_mean'][k],  analysis_stats['ent_before_std'][k],
                        ])
            except Exception as e:
                print('Failed to write CSV:', e)

            # Update progressive plots with epoch on x-axis
            try:
                epochs = np.array(analysis_stats['epoch'])
                plt.figure(figsize=(10,5))
                plt.plot(epochs, analysis_stats['norm_before_mean'], label='norm_before_mean')
                plt.fill_between(epochs,
                                 np.array(analysis_stats['norm_before_mean'])-np.array(analysis_stats['norm_before_std']),
                                 np.array(analysis_stats['norm_before_mean'])+np.array(analysis_stats['norm_before_std']),
                                 color='C0', alpha=0.15)
                plt.title('Feature L2 Norm over Epochs (before only)')
                plt.xlabel('epoch'); plt.ylabel('mean ± std')
                plt.legend(); plt.tight_layout()
                plt.savefig(join(analysis_dir, 'feature_norms_over_epochs.png'), dpi=120)
                plt.close()

                plt.figure(figsize=(10,5))
                plt.plot(epochs, analysis_stats['ent_before_mean'], label='entropy_before_mean')
                plt.fill_between(epochs,
                                 np.array(analysis_stats['ent_before_mean'])-np.array(analysis_stats['ent_before_std']),
                                 np.array(analysis_stats['ent_before_mean'])+np.array(analysis_stats['ent_before_std']),
                                 color='C0', alpha=0.15)
                plt.title('Feature Entropy over Epochs (before only)')
                plt.xlabel('epoch'); plt.ylabel('mean ± std')
                plt.legend(); plt.tight_layout()
                plt.savefig(join(analysis_dir, 'feature_entropy_over_epochs.png'), dpi=120)
                plt.close()
            except Exception as e:
                print('Failed to update progressive plots:', e)

            # SVD will be computed once per epoch after the training loop

        loss_p_value = loss_p.item()
        #mae_loss_value = mae_loss.item()
        cos_loss_value = cos_loss.item()
        loss_value = w_cos * cos_loss_value + w_p*loss_p_value


        if not math.isfinite(loss_value):
            print(loss_value)
            print(f'[rk{dist.get_rank():02d}] Loss is {loss_value}, stopping training!', flush=True)
            sys.exit(-1)
        per_loss += loss_value
        torch.cuda.synchronize()
        it += 1

    scheduler.step()
    logger.log('epoch_end_timestamps', time.time(), epoch_idx)
    epoch_loss.append(per_loss / iters_train)

    if epoch_idx == 0 or 'ema_loss' not in locals():
        ema_loss = alpha * (per_loss / iters_train) + (1 - alpha) * (per_loss / iters_train)
    else:
        ema_loss = alpha * ema_loss + (1 - alpha) * (per_loss / iters_train)

    epoch_ema_loss.append(ema_loss)

    if local_rank == 0:
        print('Epoch ', epoch_idx, ' Train AVG Loss: ', per_loss / iters_train)
        print('Epoch ', epoch_idx, ' Train EMA Loss: ', ema_loss)

    logger.log('train_losses', per_loss / iters_train, epoch_idx)
    print_to_log_file('train_loss', np.round(logger.my_fantastic_logging['train_losses'][-1], decimals=4))
    print_to_log_file(
        f"Epoch time: {np.round(logger.my_fantastic_logging['epoch_end_timestamps'][-1] - logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

    if is_ddp:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'network_weights': model_state,
        'optimizer_state': optimizer.state_dict(),
        'grad_scaler_state': None,
        'train_loss': epoch_loss,
        'current_epoch': epoch_idx
    }

    if local_rank == 0:
        torch.save(checkpoint, join(output_folder, args.model_name + '_head_latest.pt'))

        if (epoch_idx + 1) in [400, 500]:
            epoch_checkpoint_path = join(output_folder, args.model_name + f'_checkpoint_epoch_{epoch_idx+1}.pt')
            torch.save(checkpoint, epoch_checkpoint_path)

    if is_ddp:
        dist.barrier()

    # Compute and save SVD visualization once per epoch (foreground-only, input mask)
    if local_rank == 0:
        try:
            analysis_dir = join(output_folder, 'analysis_new')
            maybe_mkdir_p(analysis_dir)
            if last_student_feats is not None:
                sv_list = compute_singular_values_inputmask(last_student_feats, input_img=inp, max_samples=2048)
                if len(sv_list) > 0:
                    svd_path = join(analysis_dir, f'svd_epoch_{epoch_idx}.png')
                    plot_singular_values(sv_list, svd_path)
        except Exception as e:
            print('Failed to compute/plot SVD (epoch-level):', e)