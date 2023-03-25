#!/bin/bash

exp_folder="exps/debug_6"
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
    --loss_weight 1,100 \
    --weight_decay 1e-3 \
    --max_epochs 300 \
    --save_folder ${exp_folder}/debug \
    --amp \
    --decay_type linear \
    --step_per_epoch 200 \
    --test_frequency -1 \
    --image_shape 32,64,64 \
    --batch_size 2 \
    --num_item_nodes 2 \
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task004_ntt_debug/data_splits.pkl' \
    > ${exp_folder}/train.log &
