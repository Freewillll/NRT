#!/bin/bash

exp_folder="exps/debug_12"
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
    --max_epochs 400 \
    --save_folder ${exp_folder}/debug \
    --amp \
    --warmup_steps 200 \
    --weight_pad 0.5\
    --decay_type linear \
    --lr 1e-4 \
    --step_per_epoch 200 \
    --test_frequency 20 \
    --image_shape 32,64,64 \
    --batch_size 16 \
    --num_item_nodes 2 \
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl' \
    > ${exp_folder}/train.log &
