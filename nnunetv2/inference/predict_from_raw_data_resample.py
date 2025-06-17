import inspect
import sys
sys.path.insert(0, '/home/yoonji/AnatoMask')

from nnunetv2.training.nnUNetTrainer.variants.pretrain.STUNet import STUNet
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json

from scipy import ndimage
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import torch.nn.functional as F
import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
import argparse
from metric_evaluation import center_crop, pad_to_target_size, calculate_dice_score, calculate_nsd_score

device=torch.device("cuda")


class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_gpu: bool = True,
                 device: torch.device = torch.device(device),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            # device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!
            # why would I ever want to do that. Stupid dobby. This kills DDP inference...
            pass
        if device.type != 'cuda':
            print(f'perform_everything_on_gpu=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_gpu = False
        self.device = device
        self.perform_everything_on_gpu = perform_everything_on_gpu

    def initialize_from_trained_STUNet(self, model_training_output_dir: str,
                                             model_name:str,
                                             # checkpoint_name: str = 'checkpoint_best.pth',
                                             checkpoint_name: str = 'checkpoint_best.pth',
                                             num_classes: int = 105):
        """
        This is used when making predictions with a trained model
        
        Args:
            model_training_output_dir: Directory containing the model training output
            model_name: Name of the model
            checkpoint_name: Name of the checkpoint file
            num_classes: Number of classes in the dataset (default: 105)
        """
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        parameters = []

        checkpoint = torch.load(join(model_training_output_dir, model_name, checkpoint_name),
                                map_location=device, weights_only = False)

        trainer_name = 'STUNet_huge'
        print('Using this trainer: ', trainer_name)
        print(f'Number of classes: {num_classes}')
        inference_allowed_mirroring_axes = (0, 1, 2)

        parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration('3d_fullres')
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        #
        network = STUNet(num_input_channels, num_classes, depth=[1] * 6, dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
                      pool_op_kernel_sizes=[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,1,1]], conv_kernel_sizes=[[3, 3, 3]] * 6, enable_deep_supervision=False)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('compiling network')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder if i]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    # def predict_from_files(self,
    #                        list_of_lists_or_source_folder: Union[str, List[List[str]]],
    #                        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
    #                        save_probabilities: bool = False,
    #                        overwrite: bool = True,
    #                        num_processes_preprocessing: int = default_num_processes,
    #                        num_processes_segmentation_export: int = default_num_processes,
    #                        folder_with_segs_from_prev_stage: str = None,
    #                        num_parts: int = 1,
    #                        part_id: int = 0):
    #     """
    #     This is nnU-Net's default function for making predictions. It works best for batch predictions
    #     (predicting many images at once).
    #     """
    #     if isinstance(output_folder_or_list_of_truncated_output_files, str):
    #         output_folder = output_folder_or_list_of_truncated_output_files
    #     elif isinstance(output_folder_or_list_of_truncated_output_files, list):
    #         output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
    #     else:
    #         output_folder = None

    #     ########################
    #     # let's store the input arguments so that its clear what was used to generate the prediction
    #     if output_folder is not None:
    #         my_init_kwargs = {}
    #         for k in inspect.signature(self.predict_from_files).parameters.keys():
    #             my_init_kwargs[k] = locals()[k]
    #         my_init_kwargs = deepcopy(
    #             my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
    #         recursive_fix_for_json_export(my_init_kwargs)
    #         maybe_mkdir_p(output_folder)
    #         save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    #         # we need these two if we want to do things with the predictions like for example apply postprocessing
    #         save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
    #         save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
    #     #######################

    #     # check if we need a prediction from the previous stage
    #     if self.configuration_manager.previous_stage_name is not None:
    #         assert folder_with_segs_from_prev_stage is not None, \
    #             f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
    #             f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
    #             f' they are located via folder_with_segs_from_prev_stage'

    #     # sort out input and output filenames
    #     list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
    #         self._manage_input_and_output_lists(list_of_lists_or_source_folder,
    #                                             output_folder_or_list_of_truncated_output_files,
    #                                             folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
    #                                             save_probabilities)
    #     if len(list_of_lists_or_source_folder) == 0:
    #         return

    #     data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
    #                                                                              seg_from_prev_stage_files,
    #                                                                              output_filename_truncated,
    #                                                                              num_processes_preprocessing)

    #     return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images
        
        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )
        return pp

    # def predict_from_list_of_npy_arrays(self,
    #                                     image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
    #                                     segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
    #                                                                                                 np.ndarray,
    #                                                                                                 List[
    #                                                                                                     np.ndarray]],
    #                                     properties_or_list_of_properties: Union[dict, List[dict]],
    #                                     truncated_ofname: Union[str, List[str], None],
    #                                     num_processes: int = 3,
    #                                     save_probabilities: bool = False,
    #                                     num_processes_segmentation_export: int = default_num_processes):
    #     iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
    #                                                         segs_from_prev_stage_or_list_of_segs_from_prev_stage,
    #                                                         properties_or_list_of_properties,
    #                                                         truncated_ofname,
    #                                                         num_processes)
    #     return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    # def predict_from_data_iterator(self,
    #                                data_iterator,
    #                                save_probabilities: bool = False,
    #                                num_processes_segmentation_export: int = default_num_processes):
    #     """
    #     each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
    #     If 'ofile' is None, the result will be returned instead of written to a file
    #     """
    #     with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
    #         worker_list = [i for i in export_pool._pool]
    #         r = []
    #         for preprocessed in data_iterator:
    #             data = preprocessed['data']
    #             if isinstance(data, str):
    #                 delfile = data
    #                 data = torch.from_numpy(np.load(data))
    #                 os.remove(delfile)

    #             ofile = preprocessed['ofile']
    #             if ofile is not None:
    #                 print(f'\nPredicting {os.path.basename(ofile)}:')
    #             else:
    #                 print(f'\nPredicting image of shape {data.shape}:')

    #             print(f'perform_everything_on_gpu: {self.perform_everything_on_gpu}')

    #             properties = preprocessed['data_properties']

    #             # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
    #             # npy files
    #             proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
    #             while not proceed:
    #                 # print('sleeping')
    #                 sleep(0.1)
    #                 proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

    #             prediction = self.predict_logits_from_preprocessed_data(data).cpu()

    #             if ofile is not None:
    #                 # this needs to go into background processes
    #                 # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
    #                 #                               dataset_json, ofile, save_probabilities)
    #                 print('sending off prediction to background worker for resampling and export')
    #                 r.append(
    #                     export_pool.starmap_async(
    #                         export_prediction_from_logits,
    #                         ((prediction, properties, self.configuration_manager, self.plans_manager,
    #                           self.dataset_json, ofile, save_probabilities),)
    #                     )
    #                 )
    #             else:
    #                 # convert_predicted_logits_to_segmentation_with_correct_shape(prediction, plans_manager,
    #                 #                                                             configuration_manager, label_manager,
    #                 #                                                             properties,
    #                 #                                                             save_probabilities)
    #                 print('sending off prediction to background worker for resampling')
    #                 r.append(
    #                     export_pool.starmap_async(
    #                         convert_predicted_logits_to_segmentation_with_correct_shape, (
    #                             (prediction, self.plans_manager,
    #                              self.configuration_manager, self.label_manager,
    #                              properties,
    #                              save_probabilities),)
    #                     )
    #                 )
    #             if ofile is not None:
    #                 print(f'done with {os.path.basename(ofile)}')
    #             else:
    #                 print(f'\nDone with image of shape {data.shape}:')
    #         ret = [i.get()[0] for i in r]

    #     if isinstance(data_iterator, MultiThreadedAugmenter):
    #         data_iterator._finish()

    #     # clear lru cache
    #     compute_gaussian.cache_clear()
    #     # clear device cache
    #     empty_cache(self.device)
    #     return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
        # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
        # things a lot faster for some datasets.
        original_perform_everything_on_gpu = self.perform_everything_on_gpu
        with torch.no_grad():
            prediction = None
            if self.perform_everything_on_gpu:
                try:
                    for params in self.list_of_parameters:

                        # messing with state dict names...
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        if prediction is None:
                            prediction = self.predict_sliding_window_return_logits(data)
                        else:
                            prediction += self.predict_sliding_window_return_logits(data)

                    if len(self.list_of_parameters) > 1:
                        prediction /= len(self.list_of_parameters)

                except RuntimeError:
                    print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                          'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                    print('Error:')
                    traceback.print_exc()
                    prediction = None
                    self.perform_everything_on_gpu = False

            if prediction is None:
                for params in self.list_of_parameters:
                    # messing with state dict names...
                    if not isinstance(self.network, OptimizedModule):
                        self.network.load_state_dict(params)
                    else:
                        self.network._orig_mod.load_state_dict(params)

                    if prediction is None:
                        prediction = self.predict_sliding_window_return_logits(data)
                    else:
                        prediction += self.predict_sliding_window_return_logits(data)
                if len(self.list_of_parameters) > 1:
                    prediction /= len(self.list_of_parameters)

            print('Prediction done, transferring to CPU if needed')
            prediction = prediction.to('cpu')
            self.perform_everything_on_gpu = original_perform_everything_on_gpu
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

            num_predictons = 2 ** len(mirror_axes)
            if 0 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2,))), (2,))
            if 1 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (3,))), (3,))
            if 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (4,))), (4,))
            if 0 in mirror_axes and 1 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 3))), (2, 3))
            if 0 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 4))), (2, 4))
            if 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (3, 4))), (3, 4))
            if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
            prediction /= num_predictons
        return prediction

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose: print(f'Input shape: {input_image.shape}')
                if self.verbose: print("step_size:", self.tile_step_size)
                if self.verbose: print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                # preallocate results and num_predictions
                results_device = self.device if self.perform_everything_on_gpu else torch.device('cpu')
                if self.verbose: print('preallocating arrays')
                try:
                    data = data.to(self.device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                except RuntimeError:
                    # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                    results_device = torch.device('cpu')
                    data = data.to(results_device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                finally:
                    empty_cache(self.device)

                if self.verbose: print('running prediction')
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    workon = data[sl].unsqueeze(1)
                    workon = workon.to(self.device, non_blocking=False) # 1,1,40,224,192

                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)

                predicted_logits /= n_predictions
        empty_cache(self.device)
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

def center_crop(image, target_size):
    """
    중앙 크롭 (image: (H, W, D), target_size: (112, 112, 128))
    """
    H, W, D = image.shape
    crop_H, crop_W, crop_D = target_size

    # 중앙 기준 크롭 시작 위치 계산
    start_H = max((H - crop_H) // 2, 0)
    start_W = max((W - crop_W) // 2, 0)
    start_D = max((D - crop_D) // 2, 0)
  
    # 크롭 종료 위치 계산
    end_H = min(start_H + crop_H, H)
    end_W = min(start_W + crop_W, W)
    end_D = min(start_D + crop_D, D)

    return image[start_H:end_H, start_W:end_W, start_D:end_D]

def pad_to_target_size(image, target_size):
    """
    부족한 크기를 패딩 (image: (h, w, d), target_size: (112, 112, 128))
    """
    h, w, d = image.shape
    pad_h = max(target_size[0] - h, 0)
    pad_w = max(target_size[1] - w, 0)
    pad_d = max(target_size[2] - d, 0)

    # 앞뒤 균등하게 패딩 분배
    pad_H = (pad_h // 2, pad_h - pad_h // 2)
    pad_W = (pad_w // 2, pad_w - pad_w // 2)
    pad_D = (pad_d // 2, pad_d - pad_d // 2)

    # F.pad의 pad 순서는 (pad_left_d, pad_right_d, pad_left_w, pad_right_w, pad_left_h, pad_right_h)
    return F.pad(image, (pad_D[0], pad_D[1], pad_W[0], pad_W[1], pad_H[0], pad_H[1]), mode='constant', value=0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="for path")
    
    parser.add_argument('--inp', help='Input directory with test images')
    parser.add_argument('--checkpoint', help='Directory with checkpoint')
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--inp_label', help='Directory with ground truth labels')
    parser.add_argument('--num_classes', type=int, help='Number of classes in the dataset (default: 105)')
    
    args = parser.parse_args()
    
    print("### checkpoint_path: ", args.checkpoint)
    print("### model_name: ", args.model)

    
    # predict a bunch of files
    from nnunetv2.paths import nnUNet_results, nnUNet_raw

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
        )
    predictor.initialize_from_trained_STUNet(
        model_training_output_dir = args.checkpoint,
        model_name = args.model,
        checkpoint_name='checkpoint_best.pth',
        num_classes = args.num_classes
    )

    # predict a single numpy array
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    import os
    from datetime import datetime
    dice_scores = []
    nsd_scores =[]
    file_path = [file for file in os.listdir(args.inp)]
    
    # logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"evaluation_results_{args.model}_{timestamp}.txt"
    log_path = join(args.checkpoint,args.model, log_filename)
    
    with open(log_path, 'w') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Evaluation Results_ignore_empty_false - {args.model}\n")
        log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total Files: {len(file_path)}\n")
        log_file.write(f"Number of Classes: {args.num_classes}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.write(f"{'Index':<6}{'Filename':<30}{'Dice Score':<12}{'NSD Score':<12}\n")
        log_file.write("-" * 80 + "\n")

    print(f"📝 Logging results to: {log_path}")
    
    # predict single file iteration
    idx=0
    for file in file_path:
        img, props = SimpleITKIO().read_images([join(args.inp, file)])
        label, props = SimpleITKIO().read_images([join(args.inp_label, file.replace("_0000",""))])
        label_ = label.squeeze(0) # (112,112,128)
        #props = {'spacing': [1.0, 1.0, 1.0]}
        
        
        seg, prob = predictor.predict_single_npy_array(img, props, None, None, True)
        
        
        # to torch
        prob = torch.from_numpy(prob).float()
        label_ = torch.from_numpy(label_).long()
        del img, props, label, seg
        
        class_indices = torch.argmax(prob, dim=0) 
        pred_one_hot = F.one_hot(class_indices, num_classes=args.num_classes) 
        pred_one_hot = pred_one_hot.permute(-1, 0, 1, 2)  # (C, H, W, D)로 변환
        del prob, class_indices
        
        dice_score = calculate_dice_score(pred_one_hot, label_, args.num_classes)
        nsd_score = calculate_nsd_score(pred_one_hot, label_, args.num_classes)
        idx+=1
        
        print(idx,": dice score: ",dice_score)
        print(idx,": nsd score: ",nsd_score)
        
        with open(log_path, 'a') as log_file:
            log_file.write(f"{file}\n")
            log_file.write(f"{idx:<6}{file:<30}{dice_score:<12.4f}{nsd_score:<12.4f}\n")
            

        dice_scores.append(dice_score)
        nsd_scores.append(nsd_score)
        

        del dice_score, nsd_score, pred_one_hot
        

    avg_dice = np.mean(dice_scores)
    avg_nsd = np.mean(nsd_scores)
    std_dice = np.std(dice_scores)
    std_nsd = np.std(nsd_scores)
    
    print(f"\n{args.model}")
    print(f"\n Number of Files: {len(file_path)}")
    print(f"\n🔥 Average Dice Score: {avg_dice:.4f}")
    print(f"\n🔥 Average NSD Score: {avg_nsd:.4f}")
    
    with open(log_path, 'a') as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("FINAL STATISTICS\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Model: {args.model}\n")
        log_file.write(f"Total Files Processed: {len(file_path)}\n")
        log_file.write(f"Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}\n")
        log_file.write(f"Average NSD Score: {avg_nsd:.4f} ± {std_nsd:.4f}\n")
        log_file.write(f"Best Dice Score: {max(dice_scores):.4f}\n")
        log_file.write(f"Worst Dice Score: {min(dice_scores):.4f}\n")
        log_file.write(f"Best NSD Score: {max(nsd_scores):.4f}\n")
        log_file.write(f"Worst NSD Score: {min(nsd_scores):.4f}\n")
        log_file.write(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n✅ Results saved to: {log_path}")