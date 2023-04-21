exp_folder="exps/debug_27_ablation_decoder3"
mkdir -p $exp_folder
save_folder=${exp_folder}/evaluation

#export NUM_NODES=1
#export NUM_GPUS_PER_NODE=2
#export NODE_RANK=0
#export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=1 nohup \
python -u train.py \
    --deterministic \
    --max_epochs 300 \
    --save_folder ${save_folder} \
    --amp \
    --lr 1e-4 \
    --step_per_epoch 200 \
    --test_frequency 10 \
    --weight_pad 0.1\
    --phase 'test'\
    --seed 1025 \
    --evaluation \
    --checkpoint 'exps/debug_27_ablation_decoder3/debug/final_model.pt'\
    --image_shape '64,128,128' \
    --batch_size 1 \
    --net_config './models/configs/default_config.json'\
    --data_file '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task008_ntt_test_downsample/data_splits.pkl' \
    > ${exp_folder}/evaluation.log &