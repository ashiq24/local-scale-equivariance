#!/bin/bash
################################################################################
# DINOv2 with Registers - Adaptive Training Script for ImageNet
################################################################################
#
# This script trains an adaptive DINOv2 model with register tokens on ImageNet.
# The model uses Deep Equilibrium Models (DEM) to learn local scale equivariance.
#
# Usage:
#   bash train_dinov2_imagenet.sh [MODE] [IMAGENET_PATH] [OUTPUT_PATH]
#
# Modes:
#   fast     - Fast training with reduced DEQ iterations (dem_dinov2_fast)
#   standard - Standard training with full settings (dem_dinov2)
#
# Example:
#   bash train_dinov2_imagenet.sh fast /data/imagenet ./output/dinov2_fast
#   bash train_dinov2_imagenet.sh standard /data/imagenet ./output/dinov2_full
#  bash train_dinov2_imagenet.sh standard  /home/rahman79/Desktop/ray_ashiq/Projects/local-scale-equivariance/logs/datasets/imagenet ./logs/output/test
################################################################################

# Parse arguments
MODE="${1:-fast}"
IMAGENET_PATH="${2:-/path/to/imagenet}"
OUTPUT_PATH="${3:-./output/dinov2_${MODE}}"

# Model configuration
MODEL="vit_base_patch14_reg4_dinov2"
CONFIG_FILE="adapter_config.yaml"

# Set config based on mode
if [ "$MODE" = "fast" ]; then
    CONFIG_NAME="dem_dinov2_fast"
    EPOCHS=10
    BATCH_SIZE=64  # Increased from 32 (2x faster if memory allows)
    GRAD_ACCUM=1
elif [ "$MODE" = "standard" ]; then
    CONFIG_NAME="dem_dinov2"
    EPOCHS=10
    BATCH_SIZE=128
    GRAD_ACCUM=1
else
    echo "ERROR: Invalid mode '$MODE'. Use 'fast' or 'standard'"
    exit 1
fi

# OPTIMIZATION: More workers for faster data loading
NUM_WORKERS=24  # Increased from 16 (faster data loading)
# OPTIMIZATION: Reduce logging frequency (less I/O overhead)
LOG_INTERVAL=100  # Log every 100 batches instead of 50

# GPU settings
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

echo "============================================================"
echo "DINOv2 Adaptive Training on ImageNet"
echo "============================================================"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Config: $CONFIG_NAME"
echo "ImageNet: $IMAGENET_PATH"
echo "Output: $OUTPUT_PATH"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE (per GPU)"
echo "GPUs: $NUM_GPUS"
echo "============================================================"

# Check if ImageNet path exists
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet path not found: $IMAGENET_PATH"
    echo "Usage: bash $0 <mode> <imagenet_path> [output_path]"
    echo "Modes: fast, standard"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Set training command based on number of GPUs
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using distributed training with $NUM_GPUS GPUs"
    echo "============================================================"
    
    # Use torchrun for multi-GPU distributed training
    torchrun --nproc_per_node=$NUM_GPUS train_imagenet.py \
        --data-dir "$IMAGENET_PATH" \
        --model "$MODEL" \
        --img-size 224 \
        --pretrained \
        --num-classes 1000 \
        --our_config_file "$CONFIG_FILE" \
        --our_config_name "$CONFIG_NAME" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$NUM_WORKERS" \
        --grad-accum-steps "$GRAD_ACCUM" \
        --amp \
        --amp-dtype bfloat16 \
        --opt adamw \
        --weight-decay 0.05 \
        --sched cosine \
        --warmup-epochs 1 \
        --warmup-lr 1e-6 \
        --min-lr 1e-8 \
        --output "$OUTPUT_PATH" \
        --log-wandb \
        --pin-mem \
        --grad-checkpointing \
        --clip-grad 1.0 \
        --smoothing 0.1 \
        --log-interval "$LOG_INTERVAL" \
        --torchcompile inductor \
        --torchcompile-mode reduce-overhead
        
elif [ $NUM_GPUS -eq 1 ]; then
    echo "Using single GPU training"
    echo "============================================================"
    
    # Set minimal distributed env vars for single GPU (required by init_distributed_device)
    # These satisfy the initialization check but won't actually use distributed training
    # Using dynamic port to avoid conflicts if multiple single-GPU jobs run simultaneously
    export MASTER_ADDR=localhost
    export MASTER_PORT=$((29500 + $$ % 1000))  # $$ is the process ID, gives range 29500-30499
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
    
    python3 train_imagenet.py \
        --data-dir "$IMAGENET_PATH" \
        --model "$MODEL" \
        --img-size 224 \
        --pretrained \
        --num-classes 1000 \
        --our_config_file "$CONFIG_FILE" \
        --our_config_name "$CONFIG_NAME" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$NUM_WORKERS" \
        --grad-accum-steps "$GRAD_ACCUM" \
        --amp \
        --amp-dtype bfloat16 \
        --opt adamw \
        --weight-decay 0.05 \
        --sched cosine \
        --warmup-epochs 1 \
        --warmup-lr 1e-6 \
        --min-lr 1e-8 \
        --output "$OUTPUT_PATH" \
        --log-wandb \
        --pin-mem \
        --grad-checkpointing \
        --clip-grad 1.0 \
        --smoothing 0.1 \
        --log-interval "$LOG_INTERVAL" \
        --torchcompile inductor \
        --torchcompile-mode reduce-overhead
else
    echo "ERROR: No GPUs detected!"
    exit 1
fi

TRAIN_EXIT_CODE=$?

echo "============================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Results saved to: $OUTPUT_PATH"
echo "============================================================"

exit $TRAIN_EXIT_CODE


