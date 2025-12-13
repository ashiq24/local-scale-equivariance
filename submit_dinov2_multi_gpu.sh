#!/bin/bash
################################################################################
# SLURM Job Submission Script - Multi-GPU DINOv2 Training (DDP)
################################################################################
#
# Usage:
#   bash submit_dinov2_multi_gpu.sh [MODE] [NUM_GPUS]
#
# Modes:
#   fast     - Fast training (dem_dinov2_fast)
#   standard - Standard training (dem_dinov2)
#
# NUM_GPUS:
#   2, 4, 8  - Number of GPUs to use
#
################################################################################

# Create logs directory
mkdir -p logs

# Parse arguments
MODE="${1:-fast}"
NUM_GPUS="${2:-4}"

# Validate mode
if [[ "$MODE" != "fast" && "$MODE" != "standard" ]]; then
    echo "ERROR: Invalid mode '$MODE'. Use 'fast' or 'standard'"
    exit 1
fi

# Validate NUM_GPUS
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 2 ]; then
    echo "ERROR: NUM_GPUS must be >= 2"
    exit 1
fi

# Set config name and batch size based on mode
if [ "$MODE" = "fast" ]; then
    CONFIG_NAME="dem_dinov2_fast"
    BATCH_SIZE_PER_GPU=96
    EPOCHS=10
    TIME_LIMIT="15:55:00"
else
    CONFIG_NAME="dem_dinov2"
    BATCH_SIZE_PER_GPU=64
    EPOCHS=10
    TIME_LIMIT="23:55:00"
fi

# Calculate total batch size
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))

echo "Submitting DINOv2 ${MODE} training (${NUM_GPUS} GPUs)..."
echo "Config: ${CONFIG_NAME}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Total batch size: ${TOTAL_BATCH_SIZE}"

# Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dinov2_${MODE}_${NUM_GPUS}gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=rayyeh
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/dinov2_${MODE}_${NUM_GPUS}gpu_%j.out
#SBATCH --error=logs/dinov2_${MODE}_${NUM_GPUS}gpu_%j.err

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_NODELIST"
echo "GPUs per node: \$SLURM_GPUS_PER_NODE"
echo "Tasks per node: \$SLURM_NTASKS_PER_NODE"
echo "CPUs per task: \$SLURM_CPUS_PER_TASK"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG_NAME}"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate local_scale

# Print GPU info
nvidia-smi

# Set MASTER_PORT for distributed training
export MASTER_PORT=\$((15000 + SLURM_JOB_ID % 1000))
echo "MASTER_PORT: \$MASTER_PORT"

# Set paths (UPDATE THESE!)
IMAGENET_PATH="/path/to/imagenet"  # UPDATE THIS PATH
OUTPUT_PATH="./output/dinov2_${MODE}_${NUM_GPUS}gpu_\${SLURM_JOB_ID}"

# Check if ImageNet path exists
if [ ! -d "\$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet path not found: \$IMAGENET_PATH"
    echo "Please update IMAGENET_PATH in the script"
    exit 1
fi

# Print training configuration
echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "ImageNet: \$IMAGENET_PATH"
echo "Output: \$OUTPUT_PATH"
echo "Epochs: ${EPOCHS}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Total batch size: ${TOTAL_BATCH_SIZE}"
echo "Workers per GPU: 16"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="
echo ""

# Run distributed training with torchrun
torchrun \\
    --nproc_per_node=${NUM_GPUS} \\
    --master_port=\$MASTER_PORT \\
    train_imagenet.py \\
        --data-dir "\$IMAGENET_PATH" \\
        --model vit_base_patch14_reg4_dinov2 \\
        --img-size 224 \\
        --pretrained \\
        --our_config_file adapter_config.yaml \\
        --our_config_name ${CONFIG_NAME} \\
        --epochs ${EPOCHS} \\
        --batch-size ${BATCH_SIZE_PER_GPU} \\
        --workers 16 \\
        --amp \\
        --amp-dtype bfloat16 \\
        --opt adamw \\
        --weight-decay 0.05 \\
        --sched cosine \\
        --warmup-epochs 1 \\
        --warmup-lr 1e-6 \\
        --min-lr 1e-8 \\
        --output "\$OUTPUT_PATH" \\
        --log-wandb \\
        --pin-mem \\
        --grad-checkpointing \\
        --clip-grad 1.0 \\
        --smoothing 0.1 \\
        --torchcompile inductor \\
        --torchcompile-mode reduce-overhead

EXIT_CODE=\$?

echo ""
echo "=========================================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: \$EXIT_CODE"
fi
echo "Results: \$OUTPUT_PATH"
echo "=========================================="

exit \$EXIT_CODE
EOF

echo "Job submitted!"
echo "Monitor with: squeue -u $USER"
echo "Check logs in: logs/"
echo ""
echo "Estimated time:"
if [ "$MODE" = "fast" ]; then
    echo "  ~1.5 hours/epoch × 10 epochs = ~15 hours"
else
    echo "  ~2 hours/epoch × 10 epochs = ~20 hours"
fi

