#!/bin/bash
################################################################################
# Batch SLURM Job Submission Script - Multiple DINOv2 Configurations
################################################################################
#
# This script submits multiple training jobs with different configurations.
# Similar to your existing batch submission script but for DINOv2.
#
# Usage:
#   bash submit_dinov2_batch.sh
#
################################################################################

# Create logs directory
mkdir -p logs

# Define configurations to run
CONFIGS=(dem_dinov2_fast dem_dinov2)
NUM_GPUS=4  # Change this to 1, 2, 4, or 8 based on your cluster

echo "=========================================="
echo "Batch DINOv2 Job Submission"
echo "=========================================="
echo "Configurations: ${CONFIGS[@]}"
echo "GPUs per job: ${NUM_GPUS}"
echo "=========================================="
echo ""

# Loop through configs and submit separate jobs
for config in "${CONFIGS[@]}"; do
    
    # Set parameters based on config
    if [[ "$config" == *"fast"* ]]; then
        MODE="fast"
        BATCH_SIZE_PER_GPU=96
        TIME_LIMIT="15:55:00"
        EPOCHS=10
    else
        MODE="standard"
        BATCH_SIZE_PER_GPU=64
        TIME_LIMIT="23:55:00"
        EPOCHS=10
    fi
    
    echo "Submitting job: dinov2_${config} with ${NUM_GPUS} GPUs..."
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dinov2_${config}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=rayyeh
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/dinov2_${config}_${NUM_GPUS}gpu_%j.out
#SBATCH --error=logs/dinov2_${config}_${NUM_GPUS}gpu_%j.err

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_NODELIST"
echo "GPUs: ${NUM_GPUS}"
echo "Config: ${config}"
echo "=========================================="

# Load environment
source ~/.bashrc
squeue -u rahman79 -l

# Activate conda environment
conda activate local_scale

# Set MASTER_PORT for distributed training
export MASTER_PORT=\$((15000 + SLURM_JOB_ID % 1000))
echo "MASTER_PORT: \$MASTER_PORT"

# Print GPU info
nvidia-smi

# Set paths (UPDATE THESE!)
IMAGENET_PATH="/path/to/imagenet"  # UPDATE THIS PATH
OUTPUT_PATH="./output/dinov2_${config}_${NUM_GPUS}gpu_\${SLURM_JOB_ID}"

# Check if ImageNet path exists
if [ ! -d "\$IMAGENET_PATH" ]; then
    echo "ERROR: ImageNet path not found: \$IMAGENET_PATH"
    echo "Please update IMAGENET_PATH in this script"
    exit 1
fi

# Print training configuration
echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Model: vit_base_patch14_reg4_dinov2"
echo "Config: ${config}"
echo "ImageNet: \$IMAGENET_PATH"
echo "Output: \$OUTPUT_PATH"
echo "Epochs: ${EPOCHS}"
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Total batch size: \$((${BATCH_SIZE_PER_GPU} * ${NUM_GPUS}))"
echo "Workers per GPU: 16"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="
echo ""

# Run training with torchrun (for multi-GPU) or python (for single GPU)
if [ ${NUM_GPUS} -gt 1 ]; then
    echo "Running distributed training with torchrun..."
    
    torchrun \\
        --nproc_per_node=${NUM_GPUS} \\
        --master_port=\$MASTER_PORT \\
        train_imagenet.py \\
            --data-dir "\$IMAGENET_PATH" \\
            --model vit_base_patch14_reg4_dinov2 \\
            --img-size 224 \\
            --pretrained \\
            --our_config_file adapter_config.yaml \\
            --our_config_name ${config} \\
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
else
    echo "Running single GPU training..."
    
    python train_imagenet.py \\
        --data-dir "\$IMAGENET_PATH" \\
        --model vit_base_patch14_reg4_dinov2 \\
        --img-size 224 \\
        --pretrained \\
        --our_config_file adapter_config.yaml \\
        --our_config_name ${config} \\
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
fi

EXIT_CODE=\$?

echo ""
echo "=========================================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo "Model saved to: \$OUTPUT_PATH/model_best.pth.tar"
else
    echo "✗ Training failed with exit code: \$EXIT_CODE"
fi
echo "Results: \$OUTPUT_PATH"
echo "=========================================="

exit \$EXIT_CODE
EOF

    echo "  ✓ Job submitted: dinov2_${config}"
    sleep 1  # Small delay between submissions
    
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Monitor jobs: squeue -u $USER"
echo "Check logs: ls -lh logs/"
echo "View latest log: tail -f logs/dinov2_*_\$(squeue -u $USER -h -o %i | head -1).out"
echo "=========================================="

