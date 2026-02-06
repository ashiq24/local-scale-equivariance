#!/bin/bash
################################################################################
# Local multi-GPU training launcher for DINOv3 (torchrun)
# Usage examples:
#   bash train_dinov3_local_multi_gpu.sh dem_dinov3 --gpus 6,7,5 
#   bash train_dinov3_local_multi_gpu.sh dem_dinov3 --gpus 0,1 --batch-per-gpu 120
################################################################################

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 CONFIG_NAME [--gpus GPU_LIST] [--batch-per-gpu N] [--epochs N] [--workers N]

Positional:
  CONFIG_NAME         entry name from adapter_config.yaml (required)

Options:
  --gpus GPU_LIST     comma-separated GPU indices to use (default: all GPUs)
  --batch-per-gpu N   per-GPU batch size (default: 100)
  --epochs N          number of epochs (default: 50)
  --workers N         dataloader workers per process (default: 12)
  --data-dir PATH     ImageNet dataset path (default from script)
  --help              show this help
EOF
}

# Defaults
BATCH_PER_GPU=180
EPOCHS=30
WORKERS=10
GRAD_ACCUM=1
IMAGENET_PATH="/datasets/imagenet"
OUTPUT_ROOT=./logs/output

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

CONFIG_NAME="${1}"
shift || true

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --batch-per-gpu)
            BATCH_PER_GPU="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --data-dir)
            IMAGENET_PATH="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1"
            usage
            exit 1
            ;;
    esac
done

# If GPU_LIST not provided, detect all GPUs via nvidia-smi
if [ -z "${GPU_LIST:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -eq 0 ]; then
            echo "No GPUs detected by nvidia-smi. Set --gpus explicitly." >&2
            exit 1
        fi
        # build 0,1,2,.. list
        GPU_LIST="0"
        if [ "$GPU_COUNT" -gt 1 ]; then
            for ((i=1;i<GPU_COUNT;i++)); do
                GPU_LIST+=",${i}"
            done
        fi
    else
        echo "nvidia-smi not found; please provide --gpus list" >&2
        exit 1
    fi
fi

# Compute number of GPUs from comma-separated list
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARR[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Invalid GPU list: $GPU_LIST" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"

TOTAL_BATCH=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))
MASTER_PORT=$((15000 + $$ % 10000))
OUTPUT_PATH="${OUTPUT_ROOT}/dinov3_${CONFIG_NAME}_local_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_PATH" logs || true

echo "=========================================="
echo "Local multi-GPU DINOv3 training"
echo "=========================================="
echo "Config: ${CONFIG_NAME}"
echo "GPUs (CUDA_VISIBLE_DEVICES): ${CUDA_VISIBLE_DEVICES} (count=${NUM_GPUS})"
echo "Batch per GPU: ${BATCH_PER_GPU}  | Effective total batch: ${TOTAL_BATCH}"
echo "Epochs: ${EPOCHS}  | Workers per process: ${WORKERS}"
echo "ImageNet path: ${IMAGENET_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "=========================================="

if [ ! -d "${IMAGENET_PATH}" ]; then
    echo "ERROR: ImageNet path not found: ${IMAGENET_PATH}" >&2
    exit 1
fi

# Warm up GPU (single-process prior to torchrun)
python - <<PYCODE || { echo "GPU warm-up failed"; exit 1; }
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Visible devices:', torch.cuda.device_count())
    x = torch.zeros(1, device='cuda')
    del x
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('GPU warm-up complete')
else:
    raise SystemExit('CUDA not available')
PYCODE

# Launch distributed training using torchrun
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train_imagenet.py \
        --data-dir "${IMAGENET_PATH}" \
        --model 'vit_base_patch16_dinov3.lvd1689m' \
        --img-size 224 \
        --pretrained \
        --num-classes 1000 \
        --our_config_file adapter_config.yaml \
        --our_config_name ${CONFIG_NAME} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_PER_GPU} \
        --workers ${WORKERS} \
        --grad-accum-steps ${GRAD_ACCUM} \
        --load-weights-after-adapt '/home/rahman79/Desktop/Projects/Public_Release/local-scale-equivariance/logs/output/dinov3_dem_dinov3_local_3gpu_20260201_201730/vit_base_p_dem_dinov3_20260201-201744/model_best.pth.tar' \
        --amp \
        --amp-dtype bfloat16 \
        --opt adamw \
        --weight-decay 0.05 \
        --sched cosine \
        --warmup-epochs 2 \
        --warmup-lr 1e-8 \
        --min-lr 1e-5 \
        --output "${OUTPUT_PATH}" \
        --log-wandb \
        --pin-mem \
        --grad-checkpointing \
        --layer-decay 1.0 \
        --clip-grad 1.0 \
        --smoothing 0.1 \
        --mixup 0.1 \
        --cutmix 0.3 \
        --aa rand-m7-mstd0.5-inc1 \
        --reprob 0.25 \
        --drop-path 0.1 \
        --log-interval 100 \
        --torchcompile inductor \
        --torchcompile-mode reduce-overhead

EXIT_CODE=$?
echo "Training exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
