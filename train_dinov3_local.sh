#!/bin/bash
################################################################################
# Local GPU Training Script - DINOv3 Training
################################################################################
#
# Usage:
#   bash train_dinov3_local.sh [CONFIG_NAME]
#
# CONFIG_NAME must correspond to an entry in imagenet/config/adapter_config.yaml.
# Example: bash train_dinov3_local.sh dem_dinov3
#
################################################################################

# Parse config name
CONFIG_NAME="${1:?Please provide CONFIG_NAME (entry from adapter_config.yaml), e.g. dem_dinov3}"

# Training parameters
BATCH_SIZE=100
EPOCHS=50
GRAD_ACCUM=1
NUM_WORKERS=12
LOG_INTERVAL=100

# Set paths
IMAGENET_PATH="/home/rahman79/Desktop/ray_ashiq/Projects/local-scale-equivariance/logs/datasets/imagenet"
OUTPUT_PATH="./logs/output/dinov3_${CONFIG_NAME}_local_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_PATH"
mkdir -p logs

echo "=========================================="
echo "Local GPU Training - DINOv3"
echo "=========================================="
echo "Config: ${CONFIG_NAME}"
echo "ImageNet: ${IMAGENET_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accum steps: ${GRAD_ACCUM}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="
echo ""

# Check if ImageNet path exists
if [ ! -d "$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet path not found: $IMAGENET_PATH"
    echo "Please update IMAGENET_PATH in the script"
    exit 1
fi

# Print GPU info
nvidia-smi

# Set minimal distributed env vars for single GPU (required by init_distributed_device)
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + $$ % 1000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export CUDA_LAUNCH_BLOCKING=1

# Warm up GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    x = torch.zeros(1, device='cuda')
    del x
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('GPU warm-up complete')
" || {
    echo "ERROR: CUDA not accessible. Exiting."
    exit 1
}

# Run training
python train_imagenet.py \
    --data-dir "$IMAGENET_PATH" \
    --model 'vit_base_patch16_dinov3.lvd1689m' \
    --img-size 224 \
    --pretrained \
    --num-classes 1000 \
    --our_config_file adapter_config.yaml \
    --our_config_name ${CONFIG_NAME} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --workers ${NUM_WORKERS} \
    --grad-accum-steps ${GRAD_ACCUM} \
    --amp \
    --amp-dtype bfloat16 \
    --opt adamw \
    --weight-decay 0.05 \
    --sched cosine \
    --warmup-epochs 1 \
    --warmup-lr 1e-8 \
    --min-lr 1e-5 \
    --output "$OUTPUT_PATH" \
    --log-wandb \
    --pin-mem \
    --grad-checkpointing \
    --clip-grad 1.0 \
    --smoothing 0.1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --aa rand-m9-mstd0.5-inc1 \
    --reprob 0.25 \
    --drop-path 0.1 \
    --log-interval ${LOG_INTERVAL} \
    --torchcompile inductor \
    --torchcompile-mode reduce-overhead \

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
fi
echo "Results: $OUTPUT_PATH"
echo "=========================================="

exit $EXIT_CODE
