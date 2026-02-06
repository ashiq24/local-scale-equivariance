#!/bin/bash
################################################################################
# ImageNet zero-shot eval for SigLIP-2 with OpenAI prompt templates
# Usage:
#   ./imnet_siglip_extensive.sh dem_siglip2 --gpus 1 --checkpoint ./logs/output/siglip_dem_siglip2_local_3gpu_20260205_204357/siglip2-base-pat_dem_siglip2_20260205-204412/model_best.pth.tar
################################################################################

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 CONFIG_NAME [--checkpoint PATH] [options]

Positional:
  CONFIG_NAME            entry name from imagenet/config/adapter_config.yaml (required)

Optional:
  --checkpoint PATH      path to model checkpoint (.pth.tar). If omitted, evaluates pretrained SigLIP or untrained DEM.

Options:
  --gpus GPU_LIST        comma-separated GPU indices to use (default: all GPUs)
  --model-id HF_ID       Hugging Face SigLIP-2 model id (default: google/siglip2-base-patch16-224)
  --batch-per-gpu N      per-GPU batch size (default: 256)
  --workers N            dataloader workers per process (default: 8)
  --data-dir PATH        ImageNet dataset path (default: /datasets/imagenet)
  --prompt-set NAME      eval prompt set: openai | openai_simple | config (default: openai)
  --output PATH          output root directory (default: ./logs/output)
  --help                 show this help
EOF
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

CONFIG_NAME="${1}"
shift || true

BATCH_PER_GPU=256
WORKERS=8
SIGLIP_MODEL_ID="google/siglip2-base-patch16-224"
IMAGENET_PATH="/datasets/imagenet"
PROMPT_SET="openai"
OUTPUT_ROOT="./logs/output"
CHECKPOINT_PATH=""

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --model-id)
            SIGLIP_MODEL_ID="$2"
            shift 2
            ;;
        --batch-per-gpu)
            BATCH_PER_GPU="$2"
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
        --prompt-set)
            PROMPT_SET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
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

if [ -z "${GPU_LIST:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -eq 0 ]; then
            echo "No GPUs detected by nvidia-smi. Set --gpus explicitly." >&2
            exit 1
        fi
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

IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARR[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Invalid GPU list: $GPU_LIST" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export TOKENIZERS_PARALLELISM=false

TOTAL_BATCH=$((BATCH_PER_GPU * NUM_GPUS))
MASTER_PORT=$((15000 + $$ % 10000))
OUTPUT_PATH="${OUTPUT_ROOT}/siglip_${CONFIG_NAME}_extensive_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_PATH" logs || true

echo "=========================================="
echo "SigLIP-2 zero-shot eval (OpenAI prompts)"
echo "=========================================="
echo "Config: ${CONFIG_NAME}"
if [ -n "${CHECKPOINT_PATH}" ]; then
    echo "Checkpoint: ${CHECKPOINT_PATH}"
else
    echo "Checkpoint: (none)"
fi
echo "Model: ${SIGLIP_MODEL_ID}"
echo "Prompt set: ${PROMPT_SET}"
echo "GPUs (CUDA_VISIBLE_DEVICES): ${CUDA_VISIBLE_DEVICES} (count=${NUM_GPUS})"
echo "Batch per GPU: ${BATCH_PER_GPU} | Effective total batch: ${TOTAL_BATCH}"
echo "Workers per process: ${WORKERS}"
echo "ImageNet path: ${IMAGENET_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "=========================================="

if [ ! -d "${IMAGENET_PATH}" ]; then
    echo "ERROR: ImageNet path not found: ${IMAGENET_PATH}" >&2
    exit 1
fi

RUN_CMD=(
    train_siglip.py
    --data-dir "${IMAGENET_PATH}"
    --model "resnet50"
    --img-size 224
    --num-classes 1000
    --our_config_file adapter_config.yaml
    --our_config_name "${CONFIG_NAME}"
    --siglip-model-id "${SIGLIP_MODEL_ID}"
    --siglip-max-length 64
    --siglip-freeze-text
    --eval-metric zs_top1
    --eval-only
    --siglip-eval-prompt-set "${PROMPT_SET}"
    --batch-size ${BATCH_PER_GPU}
    --workers ${WORKERS}
    --amp
    --amp-dtype bfloat16
    --opt adamw
    --weight-decay 0.05
    --sched cosine
    --warmup-epochs 2
    --warmup-lr 1e-8
    --min-lr 1e-6
    --output "${OUTPUT_PATH}"
    --pin-mem
    --log-interval 100
)

if [ -n "${CHECKPOINT_PATH}" ]; then
    RUN_CMD+=(--load-weights-after-adapt "${CHECKPOINT_PATH}")
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${RUN_CMD[@]}"
else
    python "${RUN_CMD[@]}"
fi

EXIT_CODE=$?
echo "Eval exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
