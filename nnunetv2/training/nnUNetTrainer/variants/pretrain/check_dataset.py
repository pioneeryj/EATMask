#!/usr/bin/env python3

from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.paths import nnUNet_raw
from batchgenerators.utilities.file_and_folder_operations import join, load_json
import os

def check_dataset_integrity(dataset_id):
    dataset_name = f'Dataset{dataset_id}_FLARE22'
    dataset_path = join(nnUNet_raw, dataset_name)
    
    print(f"Checking dataset {dataset_name} at {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist")
        return
        
    dataset_json_path = join(dataset_path, 'dataset.json')
    if not os.path.exists(dataset_json_path):
        print(f"dataset.json not found at {dataset_json_path}")
        return
        
    try:
        dataset_json = load_json(dataset_json_path)
        print("Dataset JSON loaded successfully")
    except Exception as e:
        print(f"Error loading dataset.json: {str(e)}")
        return
        
    try:
        dataset = get_filenames_of_train_images_and_targets(dataset_path, dataset_json)
        print(f"Dataset has {len(dataset)} entries")
    except Exception as e:
        print(f"Error getting dataset entries: {str(e)}")
        return
    
    # Print first few keys
    print('First few keys:')
    print(list(dataset.keys())[:5])
    
    # Print first entry
    first_key = list(dataset.keys())[0]
    print('First dataset entry:')
    print(dataset[first_key])
    
    # Check for invalid paths
    issue_cases = []
    for k in dataset.keys():
        has_issues = False
        
        # Check image paths - verify if it's a list first
        images = dataset[k]['images']
        if not isinstance(images, (list, tuple)):
            print(f"Case {k} has invalid 'images' type: {type(images)}. Expected list, got {type(images).__name__}")
            print(f"Value: {images}")
            has_issues = True
            # Try to iterate anyway to see what's there
            try:
                if isinstance(images, str):
                    img_paths = [images]  # Treat as single path
                else:
                    img_paths = list(images)  # Try to convert to list
                
                for img_path in img_paths:
                    if not img_path or img_path == '/' or not os.path.isfile(img_path):
                        print(f"Case {k} has invalid image path: '{img_path}'")
            except Exception as e:
                print(f"Error checking image paths for case {k}: {str(e)}")
        else:
            # Normal case - images is a list
            for img_path in images:
                if not img_path or img_path == '/' or not os.path.isfile(img_path):
                    has_issues = True
                    print(f"Case {k} has invalid image path: '{img_path}'")
                
        # Check label path
        label_path = dataset[k]['label']
        if not label_path or label_path == '/' or not os.path.isfile(label_path):
            has_issues = True
            print(f"Case {k} has invalid label path: '{label_path}'")
            
        if has_issues:
            issue_cases.append(k)
    
    print(f'Found {len(issue_cases)} problematic cases')
    if issue_cases:
        print('Examples of problematic cases:')
        print(issue_cases[:5])
        print('Details of first problematic case:')
        print(dataset[issue_cases[0]])
    else:
        print("No problematic cases found")

if __name__ == "__main__":
    # Check Dataset 309
    check_dataset_integrity(309)
