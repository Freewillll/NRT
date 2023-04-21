#!/bin/bash

exp_folder="exps/debug_29_ablation_encoder5"
mkdir -p $exp_folder
# mkdir -p ${exp_folder}/debug


#export NUM_NODES=1
#export NUM_GPUS_PER_NODE=2
#export NODE_RANK=0
#export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=0 nohup \
python -u train.py \
    --deterministic \
    --weight_decay 1e-4 \
    --max_epochs 300 \
    --save_folder ${exp_folder}/debug \
    --amp \
    --weight_pad 0.1\
    --lr 1e-4 \
    --step_per_epoch 200 \
    --test_frequency 10 \
    --max_grad_norm 1.0 \
    --image_shape 64,128,128 \
    --seed 1025 \
    --batch_size 8 \
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl' \
    > ${exp_folder}/train.log &
