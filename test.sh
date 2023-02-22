#!/bin/bash

exp_folder="exps/exp005"
mkdir -p $exp_folder


#export NUM_NODES=1
#export NUM_GPUS_PER_NODE=2
#export NODE_RANK=0
#export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=1 nohup \
python -u train.py \
    --deterministic \
    --weight_decay 1e-4 \
    --max_epochs 200 \
    --save_folder ${exp_folder} \
    --amp \
    --step_per_epoch 200 \
    --test_frequency 3 \
    --image_shape 32,64,64 \
    --batch_size 32 \
    --num_item_nodes 16 \
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl' \
    > ${exp_folder}/train.log &
