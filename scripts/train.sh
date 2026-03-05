#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=scannet
CONFIG="None"
EXP_PATH=debug
WEIGHT="None"
RESUME=false
GPU=None
EXPERIMENT_NAME="default-experiment-name"

while getopts "p:d:c:n:w:g:r:e:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_PATH=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    e)
      EXPERIMENT_NAME=$OPTARG
      ;;
    \\?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
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

echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if [ "$RESUME" = True ]; then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"

echo " =========> RUN TASK <========="
ulimit -n 65536
if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" experiment_name="$EXPERIMENT_NAME" experiment_path="$EXP_PATH"
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT" experiment_name="$EXPERIMENT_NAME" experiment_path="$EXP_PATH"
fi