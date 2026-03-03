from encoder3D import SparseEncoder
from decoder3D import LightDecoder
from AnatoMask import SparK
import torch
from STUNet_head import STUNet

device = torch.device('cuda:1')
from torch import nn
import numpy as np
import json

import sys
sys.path.insert(0, '/home/yoonji/AnatoMask/')
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p


sys.path.append('/home/yoonji/AnatoMask/nnunetv2')
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from sklearn.model_selection import train_test_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

# 데이터 불러오기
preprocessed_dataset_folder = '/home/yoonji/nnUNet/nnUnet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans_3d_fullres'
splits_file = '/home/yoonji/nnUNet/nnUnet_preprocessed/Dataset601_organs_TotalSegmentator/splits_final.json'
splits = load_json(splits_file)
fold = 0
all_keys = splits[fold]['train']
tr_keys, val_keys = train_test_split(all_keys, test_size=0.15, random_state=42)

dataset_tr = nnUNetDataset(preprocessed_dataset_folder, tr_keys,
                           folder_with_segs_from_previous_stage=None,
                           num_images_properties_loading_threshold=0)
dataset_val = nnUNetDataset(preprocessed_dataset_folder, val_keys,
                            folder_with_segs_from_previous_stage=None,
                            num_images_properties_loading_threshold=0)

### Your nnUNet dataset json
dataset_json =load_json('/home/yoonji/nnUNet/nnUnet_preprocessed/Dataset601_organs_TotalSegmentator/dataset.json')
### Your nnUNet plans json
plans = load_json('/home/yoonji/nnUNet/nnUnet_preprocessed/Dataset601_organs_TotalSegmentator/nnUNetPlans.json')
plans_manager = PlansManager(plans)

### Your configurations
batch_size = 4
configuration_manager = plans_manager.get_configuration('3d_fullres')
label_manager = plans_manager.get_label_manager(dataset_json)
patch_size = configuration_manager.patch_size
dim = len(patch_size)
rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
initial_patch_size = get_patch_size(patch_size[-dim:],
                                    *rotation_for_DA.values(),
                                    (0.85, 1.25))
configuration_manager = plans_manager.get_configuration('3d_fullres')
label_manager = plans_manager.get_label_manager(dataset_json)

dl_tr = nnUNetDataLoader3D(dataset_tr, batch_size,
                           initial_patch_size,
                           configuration_manager.patch_size,
                           label_manager,
                           oversample_foreground_percent=0.33,
                           sampling_probabilities=None, pad_sides=None)

iters_train = len(dataset_tr) // batch_size
allowed_num_processes = get_allowed_n_proc_DA()
mt_gen_train = LimitedLenWrapper(iters_train, data_loader=dl_tr, transform=None,
                                 num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                 pin_memory= True, wait_time=0.02)
                                 
inp = next(mt_gen_train)
inp = inp['data']
print(inp.shape) # [B, C, D, H, W]

# 모델 불러오기기
class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

# STUNet_B
head = STUNet(1,1,depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512], pool_op_kernel_sizes = pool_op_kernel_sizes, conv_kernel_sizes = conv_kernel_sizes,
               enable_deep_supervision=True).to(device)
input_size = (112, 112, 128)

enc = SparseEncoder(head, input_size=input_size, sbn=False).to(device)
dec = LightDecoder(enc.downsample_ratio,sbn=False, width = 512, out_channel = 1).to(device)
model_without_ddp = SparK(
    sparse_encoder=enc, dense_decoder=dec, mask_ratio=0.6,
    densify_norm='in'
).to(device)

model = LocalDDP(model_without_ddp)

model.eval()






n_slices = 10
c, d, h, w = inp.shape[1:] # 채널, 깊이, 높이, 너비
slice_indices = torch.linspace(0, d - 1, n_slices).long()
sampled_slices = inp[:, :, slice_indices, :, :].squeeze(1)  # [B, C, D, H, W] -> [B, C, H, W] (D 제거)

# 모델에 입력 (채널과 배치를 유지한 상태로)
with torch.no_grad():
    sampled_slices = sampled_slices.to(device)  # GPU로 이동
    outputs = model(sampled_slices)  # 모델 추론 수행

# 결과 출력
print("Input shape (sampled slices):", sampled_slices.shape)  # [B, n_slices, H, W]
print("Output shape:", outputs.shape)
