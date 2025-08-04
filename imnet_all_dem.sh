#!/bin/bash

# Create logs directory if not exists
mkdir -p logs

# Define models and variants
MODELS=(vit swin deit beit)
VARIANTS=(dem)

# Loop through models and variants to submit separate jobs
for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        bash imagenet_dem.sh "$model" "${variant}_${model}" 20
    done
done
