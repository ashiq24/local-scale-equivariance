#!/bin/bash
################################################################################
# SLURM Job Submission Script - Single GPU DINOv3 Training
################################################################################
#
# Usage:
#   bash submit_dinov3_single_gpu.sh [CONFIG_NAME]
#
# CONFIG_NAME must correspond to an entry in imagenet/config/adapter_config.yaml.
#
################################################################################

# Create logs directory
mkdir -p logs

# Parse config name (no default hard-coded, you must pass one explicitly)
CONFIG_NAME="${1:?Please provide CONFIG_NAME (entry from adapter_config.yaml), e.g. dem_dinov2_v3 or a dinov3 config}"

# Training parameters
BATCH_SIZE=200
EPOCHS=20
GRAD_ACCUM=1

# OPTIMIZATION: More workers for faster data loading
NUM_WORKERS=12
# OPTIMIZATION: Reduce logging frequency (less I/O overhead)
LOG_INTERVAL=100

echo "Submitting DINOv3 training (single GPU)..."
echo "Config: ${CONFIG_NAME}"
echo "Batch size: ${BATCH_SIZE}"

# Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dinov3_${CONFIG_NAME}_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --gpus-per-node=1
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=rayyeh
#SBATCH --time=47:55:00
#SBATCH --output=logs/dinov3_${CONFIG_NAME}_1gpu_%j.out
#SBATCH --error=logs/dinov3_${CONFIG_NAME}_1gpu_%j.err
#SBATCH --exclude=h000

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_NODELIST"
echo "GPUs: \$SLURM_GPUS_PER_NODE"
echo "CPUs: \$SLURM_NTASKS"
echo "Config: \${CONFIG_NAME}"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate /home/rahman79/miniconda3/envs/local_scale

# Print GPU info
nvidia-smi

# Set paths
IMAGENET_PATH="/home/rahman79/Desktop/ray_ashiq/Projects/local-scale-equivariance/logs/datasets/imagenet"
OUTPUT_PATH="./logs/output/dinov3_${CONFIG_NAME}_1gpu_${SLURM_JOB_ID}"

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
echo "Grad accum steps: ${GRAD_ACCUM}"
echo "Workers: ${NUM_WORKERS}"
echo "Log interval: ${LOG_INTERVAL}"
echo "=========================================="
echo ""

# Set minimal distributed env vars for single GPU (required by init_distributed_device)
# These satisfy the initialization check but won't actually use distributed training
# Using dynamic port to avoid conflicts if multiple single-GPU jobs run simultaneously
export MASTER_ADDR=localhost
export MASTER_PORT=\$((29500 + \$\$ % 1000))  # $$ is the process ID, gives range 29500-30499
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
# For debugging CUDA errors
export CUDA_LAUNCH_BLOCKING=1

# Wait for GPU to be fully available (helps with race conditions on shared clusters)
sleep 5

# Warm up GPU with a small allocation to ensure it's ready
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    # Warm up: allocate and free a small tensor to initialize CUDA context
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
python train_imagenet.py \\
    --data-dir "\$IMAGENET_PATH" \\
    --model 'vit_base_patch16_dinov3.lvd1689m' \\
    --img-size 224 \\
    --pretrained \\
    --num-classes 1000 \\
    --our_config_file adapter_config.yaml \\
    --our_config_name ${CONFIG_NAME} \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --workers ${NUM_WORKERS} \\
    --grad-accum-steps ${GRAD_ACCUM} \\
    --amp \\
    --amp-dtype bfloat16 \\
    --opt adamw \\
    --weight-decay 0.05 \\
    --sched cosine \\
    --warmup-epochs 2 \\
    --warmup-lr 1e-8 \\
    --min-lr 1e-8 \\
    --output "\$OUTPUT_PATH" \\
    --log-wandb \\
    --pin-mem \\
    --grad-checkpointing \\
    --clip-grad 1.0 \\
    --smoothing 0.1 \\
    --log-interval ${LOG_INTERVAL} \\
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
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: logs/"


