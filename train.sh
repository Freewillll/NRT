#!/bin/bash

exp_folder="exps/exp001"
mkdir -p $exp_folder
#CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --deterministic --max_epochs 50 --save_folder ${exp_folder} --amp > ${exp_folder}/fullsize_adam.log &


export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=0,1 nohup \
python -u -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    train.py \
    --deterministic \
    --max_epochs 100 \
    --save_folder ${exp_folder} \
    --amp \
    --step_per_epoch 200 \
    --test_frequency 3 \
    --image_shape 32,64,64 \
    --batch_size 16 \
    --num_item_nodes 16 \
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl' \
    > ${exp_folder}/train.log & 
