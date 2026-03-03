#!/bin/bash

##############################################################################
# STUNetVarianceTrainer Training Script
#
# Trains STUNet with Variance Loss for uncertainty estimation
# Usage:
#   bash train_variance_loss.sh                    # Default settings
#   bash train_variance_loss.sh --fold 1 --var-weight 0.5 --gpus 4
##############################################################################

set -e

# ===== Configuration =====
DATASET="Dataset606_all_TotalSegmentator"
CONFIGURATION="3d_fullres"
TRAINER="STUNetVarianceTrainer"
FOLD=0
VAR_WEIGHT=0.5
NUM_GPUS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --var-weight)
            VAR_WEIGHT="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash train_variance_loss.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fold FOLD              Cross-validation fold (default: 0)"
            echo "  --var-weight WEIGHT      Variance loss weight (default: 0.5)"
            echo "  --gpus NUM               Number of GPUs (default: 1)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash train_variance_loss.sh"
            echo "  bash train_variance_loss.sh --fold 1 --var-weight 0.5"
            echo "  bash train_variance_loss.sh --gpus 4 --var-weight 1.0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ===== Print Configuration =====
echo "=========================================="
echo "STUNetVarianceTrainer Training"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Configuration: $CONFIGURATION"
echo "Trainer: $TRAINER"
echo "Fold: $FOLD"
echo "Variance Loss Weight: $VAR_WEIGHT"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# ===== Execute Training =====
cd /home/yoonji/AnatoMask

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Starting single GPU training..."
    python -m nnunetv2.run.run_training \
        "$DATASET" \
        "$CONFIGURATION" \
        "$FOLD" \
        "$TRAINER" \
        --var_weight "$VAR_WEIGHT"

else
    # Multi-GPU training with DDP
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node="$NUM_GPUS" \
        -m nnunetv2.run.run_training \
        "$DATASET" \
        "$CONFIGURATION" \
        "$FOLD" \
        "$TRAINER" \
        --var_weight "$VAR_WEIGHT"
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
