#!/bin/bash
################################################################################
# SLURM Job Submission Script - Single GPU DINOv2 Training
################################################################################
#
# Usage:
#   bash submit_dinov2_single_gpu.sh [MODE]
#
# Modes:
#   fast     - Fast training (dem_dinov2_fast)
#   standard - Standard training (dem_dinov2)
#
################################################################################

# Create logs directory
mkdir -p logs

# Parse mode (default: fast)
MODE="${1:-fast}"

# Validate mode
if [[ "$MODE" != "fast" && "$MODE" != "standard" ]]; then
    echo "ERROR: Invalid mode '$MODE'. Use 'fast' or 'standard'"
    exit 1
fi

# Set config name based on mode
if [ "$MODE" = "fast" ]; then
    CONFIG_NAME="dem_dinov2_fast"
    BATCH_SIZE=32
    EPOCHS=10
else
    CONFIG_NAME="dem_dinov2"
    BATCH_SIZE=24
    EPOCHS=10
fi

echo "Submitting DINOv2 ${MODE} training (single GPU)..."
echo "Config: ${CONFIG_NAME}"
echo "Batch size: ${BATCH_SIZE}"

# Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dinov2_${MODE}_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=rayyeh
#SBATCH --time=47:55:00
#SBATCH --output=logs/dinov2_${MODE}_1gpu_%j.out
#SBATCH --error=logs/dinov2_${MODE}_1gpu_%j.err

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_NODELIST"
echo "GPUs: \$SLURM_GPUS_PER_NODE"
echo "CPUs: \$SLURM_NTASKS"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG_NAME}"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate /home/rahman79/miniconda3/envs/local_scale

# Print GPU info
nvidia-smi

# Set paths (UPDATE THESE!)
IMAGENET_PATH="/home/rahman79/Desktop/ray_ashiq/Projects/local-scale-equivariance/logs/datasets/imagenet"  # UPDATE THIS PATH
OUTPUT_PATH="./logs/output/dinov2_${MODE}_1gpu_\${SLURM_JOB_ID}"

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
echo "Batch size: ${BATCH_SIZE}"
echo "Workers: 16"
echo "=========================================="
echo ""

# Run training
python train_imagenet.py \\
    --data-dir "\$IMAGENET_PATH" \\
    --model vit_base_patch14_reg4_dinov2 \\
    --img-size 224 \\
    --pretrained \\
    --our_config_file adapter_config.yaml \\
    --our_config_name ${CONFIG_NAME} \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
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

