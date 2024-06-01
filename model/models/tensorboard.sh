#! /bin/bash

module load gcc python openmpi py-tensorflow

ipnport=$(shuf -i8000-9999 -n1)

TENSOR_DIR=$1
tensorboard --logdir ${TENSOR_DIR}  --port=${ipnport} --bind_all
