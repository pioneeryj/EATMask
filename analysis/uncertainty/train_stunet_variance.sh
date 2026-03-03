#!/bin/bash

##############################################################################
# STUNet with Aleatoric Uncertainty Training Script
#
# 105-class segmentation with variance loss
# Supports both single GPU and multi-GPU (DDP) training
#
# Usage:
#   # Single GPU
#   bash train_stunet_variance.sh
#
#   # Multi-GPU (4 GPUs)
#   bash train_stunet_variance.sh --gpus 4
#
##############################################################################

set -e  # Exit on error

# ===== Configuration =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLD=0
EPOCHS=100
BATCH_SIZE=2
LEARNING_RATE=1e-4
WARMUP_EPOCHS=20
NUM_GPUS=1
MODEL_NAME="stunet_aleatoric"
OUTPUT_DIR="/nas_homes/yoonji/AnatoMask/nnunet_trained_models/stunet_aleatoric_variance"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: bash train_stunet_variance.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fold FOLD              Fold number (default: 0)"
            echo "  --epochs EPOCHS          Number of epochs (default: 100)"
            echo "  --batch-size SIZE        Batch size per GPU (default: 4)"
            echo "  --lr LR                  Learning rate (default: 1e-4)"
            echo "  --warmup EPOCHS          Warmup epochs (default: 20)"
            echo "  --gpus NUM               Number of GPUs (default: 1)"
            echo "  --model-name NAME        Model name (default: stunet_aleatoric)"
            echo "  --output DIR             Output directory"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Single GPU training"
            echo "  bash train_stunet_variance.sh --epochs 100"
            echo ""
            echo "  # Multi-GPU training (4 GPUs)"
            echo "  bash train_stunet_variance.sh --gpus 4 --batch-size 16 --epochs 50"
            echo ""
            exit 0
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_EPOCHS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ===== Environment Setup =====
echo "=========================================="
echo "STUNet Variance Loss Training"
echo "=========================================="
echo "Fold: $FOLD"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Warmup Epochs: $WARMUP_EPOCHS"
echo "Number of GPUs: $NUM_GPUS"
echo "Model Name: $MODEL_NAME"
echo "=========================================="

# Change to script directory
cd "$SCRIPT_DIR"

# Python path
PYTHON_SCRIPT="$SCRIPT_DIR/nnunetv2/training/nnUNetTrainer/variants/pretrain/train_stunet_aleatoric.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Training script not found at $PYTHON_SCRIPT"
    exit 1
fi

# ===== Build Command =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD="python $PYTHON_SCRIPT"
CMD="$CMD --fold $FOLD"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --lr $LEARNING_RATE"
CMD="$CMD --warmup $WARMUP_EPOCHS"
CMD="$CMD --model_name $MODEL_NAME"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

# ===== Execute Training =====
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo ""
    echo "Starting single GPU training..."
    echo "Command: $CMD"
    echo ""

    eval "$CMD"

else
    # Multi-GPU training with DDP
    echo ""
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    echo ""

    DDP_CMD="torchrun --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT"
    DDP_CMD="$DDP_CMD --fold $FOLD"
    DDP_CMD="$DDP_CMD --epochs $EPOCHS"
    DDP_CMD="$DDP_CMD --batch_size $BATCH_SIZE"
    DDP_CMD="$DDP_CMD --lr $LEARNING_RATE"
    DDP_CMD="$DDP_CMD --warmup $WARMUP_EPOCHS"
    DDP_CMD="$DDP_CMD --model_name $MODEL_NAME"

    if [ -n "$OUTPUT_DIR" ]; then
        DDP_CMD="$DDP_CMD --output $OUTPUT_DIR"
    fi

    echo "Command: $DDP_CMD"
    echo ""

    eval "$DDP_CMD"
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
