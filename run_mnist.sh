CONFIG_FILE=$1
CONFIG_NAME=$2
RANDOM_SEED=$3

python train_mnist.py \
  --config_file "${CONFIG_FILE}" \
  --config "${CONFIG_NAME}" \
  --random_seed "${RANDOM_SEED}"
