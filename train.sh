#!/usr/bin/env bash

GPU_ID=${1-"1"}
export WANDB_ENTITY="clif"
export WANDB_PROJECT="adapters"

CUDA_VISIBLE_DEVICES=$GPU_ID nohup python qlora_train.py > train_${GPU_ID}.out 2>&1 &
