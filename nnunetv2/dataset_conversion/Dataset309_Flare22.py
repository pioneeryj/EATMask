import os
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json, generate_dataset_json_flare

base_path = "/nas_homes/yoonji/medmask"

os.environ['nnUNet_raw'] = os.path.join(base_path, "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = os.path.join(base_path, "nnUNet_preprocessed")
os.environ['nnUNet_results'] = os.path.join(base_path, "nnUNet_results")
from nnunetv2.paths import nnUNet_raw


def convert_flare22(nnunet_dataset_id: int = 309):
    """
    기존에 준비된 nnU-Net 형식 데이터에 대해 dataset.json 파일을 생성합니다.
    데이터는 이미 imagesTr, labelsTr 폴더에 적절한 형식으로 저장되어 있어야 합니다.
    """
    task_name = "FLARE22"
    
    # nnU-Net 폴더명 생성
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)
    
    # 데이터셋 기본 경로
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    
    # 폴더 존재 확인
    if not os.path.exists(imagestr):
        raise RuntimeError(f"Training images folder not found: {imagestr}")
    if not os.path.exists(labelstr):
        raise RuntimeError(f"Training labels folder not found: {labelstr}")
    
    # 훈련 이미지 파일 목록 가져오기 (nii.gz 파일만)
    training_images = subfiles(imagestr, suffix=".nii.gz", join=False, sort=True)
    training_labels = subfiles(labelstr, suffix=".nii.gz", join=False, sort=True)
    
    print(f"Found {len(training_images)} training images")
    print(f"Found {len(training_labels)} training labels")
    
    # 파일 개수 확인
    if len(training_images) != len(training_labels):
        print(f"Warning: Number of images ({len(training_images)}) doesn't match number of labels ({len(training_labels)})")
    
    # FLARE22 데이터셋의 라벨 정의 (13개 클래스)
    # 실제 데이터셋에 맞게 수정이 필요할 수 있습니다
    labels = {
        "background": 0,
        "liver": 1,
        "right kidney": 2,  
        "spleen": 3,
        "pancreas": 4,
        "aorta": 5,
        "inferior vena cava": 6,
        "right adrenal gland": 7,
        "left adrenal gland": 8,
        "gallbladder": 9,
        "esophagus": 10,
        "stomach": 11,
        "duodenum": 12,
        "left kidney": 13
    }
    
    # dataset.json 생성
    generate_dataset_json_flare(
        out_base, 
        channel_names={0: "CT"},  # FLARE22는 CT 이미지
        labels=labels,
        num_training_cases=len(training_images), 
        file_ending='.nii.gz',
        dataset_name=task_name, 
        reference='https://flare22.grand-challenge.org/',
        release='https://zenodo.org/record/7155725',
        overwrite_image_reader_writer='NibabelIOWithReorient',
        description="FLARE22: Fast and Low GPU memory Abdominal oRgan sEgmentation challenge. "
                   "Multi-organ segmentation in abdominal CT scans with 13 organ classes. "
                   "Dataset converted for nnU-Net usage.",
                   imagestr = imagestr
    )
    
    print(f"Successfully generated dataset.json for {task_name}")
    print(f"Dataset location: {out_base}")
    print(f"Number of training cases: {len(training_images)}")
    print(f"Number of classes: {len(labels)}")


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_id', required=False, type=int, default=309, 
                       help='nnU-Net Dataset ID, default: 309')
    args = parser.parse_args()
    
    convert_flare22(args.dataset_id)