#!/bin/bash

# SimpleUNet with Aleatoric Uncertainty Training Script

# 기본 설정
FOLD=0
EPOCHS=100
BATCH_SIZE=4
LR=1e-4
WARMUP=20
MODEL_NAME="simpleunet_aleatoric"
OUTPUT_FOLDER="/nas_homes/yoonji/AnatoMask/aleatoric_kendalgal/simpleunet_aleatoric"
# 선택적 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 로깅
echo "=========================================="
echo "SimpleUNet Aleatoric Uncertainty Training"
echo "=========================================="
echo "Fold: $FOLD"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "Warmup Epochs: $WARMUP"
echo "Model Name: $MODEL_NAME"
if [ ! -z "$OUTPUT_FOLDER" ]; then
    echo "Output Folder: $OUTPUT_FOLDER"
fi
echo "=========================================="
echo ""

# Python 실행
cd /home/yoonji/AnatoMask

CUDA_VISIBLE_DEVICES=0 python -u nnunetv2/training/nnUNetTrainer/variants/pretrain/train_stunet_aleatoric.py \
    --fold $FOLD \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup $WARMUP \
    --model_name $MODEL_NAME \
    ${OUTPUT_FOLDER:+--output $OUTPUT_FOLDER}

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
