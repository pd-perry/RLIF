#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=:/home/ubuntu/itv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_API_KEY=0770b23a4c7e8a79bd117f262b7b8b6f38cbc11a
export D4RL_SUPPRESS_IMPORT_ERROR=1


export N_GPUS=$(nvidia-smi -L | wc -l)
export N_JOBS=18


parallel -j $N_JOBS --linebuffer --delay 1 \
    'CUDA_VISIBLE_DEVICES=$(({%} % 4)) 'python3.8 -m RLIF.examples.train_rlif_main \
           --seed={1} \
            --project_name='{2}' \
            --env_name='{3}' \
            --sparse_env={4} \
            --dataset_dir={5} \
            --offline_ratio={6} \
            --eval_episodes=100 \
            --log_interval=1000 \
            --eval_interval=10000 \
            --max_traj_length={7} \
            --utd_ratio={8} \
            --ground_truth_agent_dir={9} \
            --intervene_threshold={10} \
            --intervention_strategy={11} \
            --expert_dir={12} \
            --save_model=True \
        ::: 24 42 43 \
        ::: 'pen-value-based-rlif' \
        ::: "pen-expert-v1" \
        ::: 'AdroitHandPenSparse-v1' \
        ::: './RLIF/iclr_datasets/pen-expert-v1_50trajs' \
        ::: 0.0 \
        ::: 200 \
        ::: 5 \
        ::: './RLIF/experts/rlpd_experts/s24_0pretrain_15utd_0.3offline_LN/model.pkl' \
        ::: 0.97 \
        ::: 'q' \
        ::: './RLIF/experts/bc_experts/960d7aba1a654a6aae323b7558bf3378/model.pkl' './RLIF/experts/bc_experts/382bae4018ad468dae384799ab8d81ba/model.pkl' './RLIF/experts/bc_experts/9762b4feeac74df8847500f0e2869aee/model.pkl' \


