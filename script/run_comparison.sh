#!/bin/bash

# Uncertainty Methods 비교 스크립트 실행

cd /home/yoonji/AnatoMask

# 선택적 인자 파싱
FOLD=0
NUM_SAMPLES=10
MODEL_PATH="/nas_homes/yoonji/AnatoMask/aleatoric_kendalgal/simpleunet_aleatoric/simpleunet_aleatoric_best.pt"
OUTPUT_FOLDER="/home/yoonji/AnatoMask/aleatoric_kendal_comparison"
NUM_SAMPLES=10
while [[ $# -gt 0 ]]; do
    case $1 in
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
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

echo "=========================================="
echo "Uncertainty Methods Comparison"
echo "Variance-based vs Entropy-based"
echo "=========================================="
echo "Fold: $FOLD"
echo "Number of samples: $NUM_SAMPLES"
if [ ! -z "$MODEL_PATH" ]; then
    echo "Model path: $MODEL_PATH"
fi
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=1 python -u compare_uncertainty_methods.py \
    --fold $FOLD \
    --num_samples $NUM_SAMPLES \
    ${MODEL_PATH:+--model_path $MODEL_PATH} \
    ${OUTPUT_FOLDER:+--output $OUTPUT_FOLDER}

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
