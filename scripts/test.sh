# Set working dir
cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TEST_CODE=test.py

DATASET=scannet
CONFIG="None"
EXP_PATH=debug
WEIGHT=model_best
GPU=None
EXPERIMENT_NAME="default-experiment-name"

while getopts "p:d:c:n:w:g:e:" opt; do
  case $opt in
    p) PYTHON=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    n) EXP_PATH=$OPTARG ;;
    w) WEIGHT=$OPTARG ;;
    g) GPU=$OPTARG ;;
    e) EXPERIMENT_NAME=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]; then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_PATH"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"
echo "Experiment Name: $EXPERIMENT_NAME"

EXP_DIR=${EXP_PATH}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code

if [ "${CONFIG}" = "None" ]; then
  CONFIG_DIR=${EXP_DIR}/config.py
else
  CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=$CODE_DIR

echo " =========> RUN TASK <========="
ulimit -n 65536
$PYTHON -u tools/$TEST_CODE \
  --config-file "$CONFIG_DIR" \
  --num-gpus "$GPU" \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}/${WEIGHT}.pth" experiment_name="$EXPERIMENT_NAME" experiment_path="$EXP_PATH"