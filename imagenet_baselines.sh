#!/bin/bash

# Validate input arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <model> <config_name> <epochs>"
    exit 1
fi

model="$1"
config_name="$2"
epochs="$3"

# training dataset choice 
debug_mode=false  # Set to true to enable sample limits
train_dataset='scale_imagenet' # 'original_imagenet' or 'scale_imagenet'

# Common configurations
dataset_path="/depot/rayyeh/data/CAYang/datasets/imagenet/"
multi_scale_datapath="../../image_datasets/scale_imagenet_2/"
batch_size=128
wandb_project="imagenet-adaptation"

# Debug parameters
debug_params=""
if [ "$debug_mode" = true ]; then
    debug_params="--train-num-samples 100 --val-num-samples 50"
fi

# generate random port
PORT=$((RANDOM % 10000 + 10000))
# print port number 
echo "**************Port********: $PORT"
# Base command components

output_dir="/scratch/gautschi/rahman79/output/"
run_command=(
    torchrun --nproc_per_node=1 --nnodes=1 --master_port="$PORT"
)
echo "Output directory: $output_dir"

scale_training_params=""
if [ "$train_dataset" = 'scale_imagenet' ]; then
    scale_training_params="--train-num-samples 10000 --class-map imagenet_synsets.txt"
    dataset_path="$multi_scale_datapath"
fi

echo "Dataset path: $dataset_path"
echo "Multi-scale dataset path: $multi_scale_datapath"

base_cmd=(
    train_imagenet.py
    --data-dir "$dataset_path"
    --multi_scale_datapath "$multi_scale_datapath"
    --train-split train
    --val-split val
    --multi_scale_val_split val
    --warmup-lr 1e-8
    --opt adamw
    --output "$output_dir"
    --sched cosine
    --weight-decay 0.05
    --warmup-epochs 5
    --aa original
    --mixup 0.8
    --cutmix 1.0
    --no-prefetcher
    --smoothing 0.1
    --reprob 0.25
    --amp
    --clip-grad 5.0
    --multi_scale_class_map imagenet_synsets.txt
    -j 20
    --layer-decay 0.75
    --batch-size "$batch_size"
    --pretrained
    --our_config_file adapter_config.yaml
    --our_config_name "$config_name"
    --log-wandb
    --wandb-project "$wandb_project"
    --epochs "$epochs"
    --do_pretrain_eval
    --per_scale_eval_freq 10
)

# Model-specific configurations
case "$model" in
    swin)
        model_params=(
            --model swin_tiny_patch4_window7_224
            --layer-decay 0.75
            --drop-path 0.2
        )
        ;;
    vit)
        model_params=(
            --model vit_little_patch16_reg4_gap_256.sbb_in1k
            --layer-decay 0.75
            --drop-path 0.1
        )
        ;;
    deit)
        model_params=(
            --model deit_tiny_patch16_224.fb_in1k
            --layer-decay 0.75
            --drop-path 0.1
        )
        ;;
    beit)
        model_params=(
            --model beitv2_base_patch16_224
            --layer-decay 0.85
            --drop-path 0.1
        )
        ;;
    *)
        echo "Invalid model: $model. Valid options: swin, vit, deit, beit"
        exit 1
        ;;
esac

# Combine and run command
full_cmd=("${run_command[@]}" "${base_cmd[@]}" "${model_params[@]}" $debug_params $scale_training_params)
echo "Running command:"
printf "%s " "${full_cmd[@]}"
echo
"${full_cmd[@]}"
