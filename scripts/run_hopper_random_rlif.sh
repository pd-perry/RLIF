#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=:/home/ubuntu/itv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY=0770b23a4c7e8a79bd117f262b7b8b6f38cbc11a
export D4RL_SUPPRESS_IMPORT_ERROR=1


export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=32

EXP_NAME='rlpd-experts-gauss'
OUTPUT_DIR="./experiment_output"

parallel -j $N_JOBS --linebuffer --delay 1 \
    'CUDA_VISIBLE_DEVICES=$(({%} % 4)) 'python3.8 -m RLIF.examples.train_rlif_random_main \
            --seed={1} \
            --project_name='{2}' \
            --env_name='{3}' \
            --sparse_env={4} \
            --dataset_dir={5} \
            --offline_ratio={6} \
            --eval_episodes=50 \
            --log_interval=1000 \
            --eval_interval=10000 \
            --max_traj_length={7} \
            --utd_ratio={8} \
            --ground_truth_agent_dir={9} \
            --intervene_threshold={10} \
            --intervention_strategy={11} \
            --expert_dir={12} \
            --intervention_rate=1 \
            --save_model=False \
        ::: 24 42 43 \
        ::: 'hopper-random-rlif' \
        ::: "hopper-expert-v2" \
        ::: 'Hopper-v2' \
        ::: './RLIF/iclr_datasets/hopper-expert-v2_50trajs' \
        ::: 0.0 \
        ::: 1000 \
        ::: 15 \
        ::: './RLIF/experts/rlpd_experts/s24_hopper-expert-v2env/model.pkl' \
        ::: 0.975 \
        ::: 'unif' \
        ::: './RLIF/experts/rlpd_experts/s24_hopper-expert-v2env/model.pkl' './RLIF/experts/bc_experts/b750e30dc83a4158a7ded9ccc8ef3298/model.pkl' './RLIF/experts/bc_experts/6d31b3a8d5d049dfa08676fb6fc8eb20/model.pkl' \


