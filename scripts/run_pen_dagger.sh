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



EXP_NAME='pen-dagger-uniform'
OUTPUT_DIR="./experiment_output"

parallel -j $N_JOBS --linebuffer --delay 1 \
    'CUDA_VISIBLE_DEVICES=$(({%} % 4)) 'python3.8 -m RLIF.examples.train_dagger_main \
            --seed={1} \
            --sparse_env='{2}' \
            --dense_env='{3}' \
            --dataset_dir={4} \
            --expert_dir={5} \
            --pretrain_n_epochs=500 \
            --pretrain_n_train_step_per_epoch=200 \
            --n_iters=100 \
            --n_epochs=25 \
            --n_train_step_per_epoch=100 \
            --max_traj_length=200 \
            --collect_n_trajs=5 \
            --eval_n_trajs=50 \
            --intervention_strategy={6} \
            --intervention_rate={7} \
            --intervene_threshold={8} \
            --compare_optimal={9} \
            --ground_truth_agent_dir={10} \
            --train_type='bc' \
            --cql.cql_min_q_weight=0 \
            --cql.policy_weight_decay={11} \
            --logging.output_dir="$OUTPUT_DIR/$EXP_NAME" \
            --logging.online=True \
            --logging.prefix='' \
            --logging.entity 'perrydong' \
            --logging.project="$EXP_NAME" \
            --logging.random_delay=1.0 \
        ::: 24 42 43 \
        ::: 'AdroitHandPenSparse-v1' \
        ::: "pen-human-v1"\
        ::: './RLIF/iclr_datasets/pen-expert-v1_50trajs' \
        ::: './RLIF/experts/bc_experts/960d7aba1a654a6aae323b7558bf3378/model.pkl' './RLIF/experts/bc_experts/382bae4018ad468dae384799ab8d81ba/model.pkl' './RLIF/experts/bc_experts/9762b4feeac74df8847500f0e2869aee/model.pkl' \
        ::: 'dagger' \
        :::+ 1 \
        :::+ 0 \
        :::+ False \
        ::: './RLIF/experts/rlpd_experts/s24_0pretrain_15utd_0.3offline_LN/model.pkl' \
        ::: 1e-3 \


        