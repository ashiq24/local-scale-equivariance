#!/bin/bash

CONFIG_FILE="config_1.yaml"

data_aug_configs=(
    swin_data_augmentation
    res_net_data_augmentation
    vit_data_augmentation
    dino_data_augmentation
    beit_data_augmentation
    deit_data_augmentation
)

equiv_configs=(
    swin_equiv
    res_net_equiv
    vit_equiv
    dino_equiv
    beit_equiv
    deit_equiv
)

canon_configs=(
    swin_cannonicalizion
    res_net_cannonicalization
    vit_cannonicalization
    dino_cannonicalization
    beit_cannonicalization
    deit_cannonicalization
)

ada_dem_configs=(
    ada_swin_dem
    ada_res_net_dem_1
    ada_vit_dem_1
    ada_dino_dem_1
    ada_beit_dem
    ada_deit_dem_1
)

CONFIGS_TO_RUN=$data_aug_configs

for random_seed in {70..72}; do
    for cfg in "${CONFIGS_TO_RUN[@]}"; do
        bash run_mnist.sh "${CONFIG_FILE}" "${cfg}" "${random_seed}"
    done
done
