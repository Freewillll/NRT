#!/bin/bash

exp_folder="exps/debug_22"
mkdir -p $exp_folder
# mkdir -p ${exp_folder}/debug


#export NUM_NODES=1
#export NUM_GPUS_PER_NODE=2
#export NODE_RANK=0
#export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=1 nohup \
python -u train.py \
    --deterministic \
    --weight_decay 1e-4 \
    --max_epochs 350 \
    --save_folder ${exp_folder}/debug \
    --amp \
    --weight_pad 0.1\
    --lr 1e-4 \
    --step_per_epoch 200 \
    --test_frequency 10 \
    --max_grad_norm 1.0 \
    --image_shape 32,64,64 \
    --seed 1025 \
    --batch_size 32 \
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl' \
    > ${exp_folder}/train.log &
