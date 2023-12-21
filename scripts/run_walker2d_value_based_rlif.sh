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


parallel -j $N_JOBS --linebuffer --delay 1 \
    'CUDA_VISIBLE_DEVICES=$(({%} % 4)) 'python3.8 -m intervene.rlpd_itv_main \
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
            --save_model=False \
        ::: 24 42 43 \
        ::: 'walker2d-value-based-rlif' \
        ::: "walker2d-expert-v2" \
        ::: 'Walker2d-v2' \
        ::: './RLIF/iclr_datasets/walker2d-expert-v2_10trajs' \
        ::: 0.0 \
        ::: 1000 \
        ::: 1 \
        ::: './RLIF/experts/bc_experts/461a33ca2a8c48f497acfce6bf4eb0e7/model.pkl' \
        ::: 0.99 \
        ::: 'q' \
        ::: './RLIF/experts/bc_experts/461a33ca2a8c48f497acfce6bf4eb0e7/model.pkl' './RLIF/experts/bc_experts/59456c494a8a4ff988178c1524e36298/model.pkl' './RLIF/experts/bc_experts/9cd6157b007942ac95d8d8d870b38941/model.pkl' \


