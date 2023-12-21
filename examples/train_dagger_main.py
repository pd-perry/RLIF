import os
import time

import numpy as np
from pprint import pprint

import numpy as np
import distrax

import tqdm
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from ml_collections import config_flags

import gym
import gymnasium
import d4rl

import absl.app
from absl import flags

from ..agents.conservative_sac import ConservativeSAC
from ..agents.iql import IQLLearner, get_iql_policy_from_model
from ..agents.rlpd import get_rlpd_policy_from_model
from ..agents.dagger_based_learner import DAggerBasedLearner
from ..utils.env_utils import GymnasiumWrapper
from ..models.model import RandomSamplerPolicy, TanhGaussianPolicy, FullyConnectedQFunction, get_policy_from_model, load_model
from ..utils.utils import define_flags_with_default, set_random_seed, get_user_flags, WandBLogger

FLAGS = flags.FLAGS

FLAGS_DEF = define_flags_with_default(
            dense_env='pen-human-v1',
            sparse_env='AdroitHandPenSparse-v1',
            dataset_dir='',
            expert_dir='./intervene/bc_output/grace_bc_adroit_d4rl_policies/994fb9e340c94a2aa94a9685bb64b3ae/model.pkl',
            seed=24,
            task_reward=False,

            pretrain_n_epochs=1,
            pretrain_n_train_step_per_epoch=200,

            n_iters=10,
            n_epochs=1,
            n_train_step_per_epoch=200,
            max_traj_length=200,
            collect_n_trajs=5,
            batch_size=256,
            rl_reward_multiplier=1,
            eval_n_trajs=1,

            intervention_rate=0,
            intervention_strategy='',
            ground_truth_agent_dir='./intervene/grace_iql_expert_adroit/0542bcee61f74bd783c58ef86ed01316/model.pkl',
            intervene_threshold=0.0,
            intervene_n_steps=4,
            compare_optimal=True,
            intervene_temperature=0.0,

            policy_weight_decay=0.0,
            iql_bc_loss_weight=0,
            iql_expectile=0.9,
            iql_temperature=0.1,
            iql_log_stds=0.0,

            rlpd_offline_ratio=0.5,
            rlpd_utd_ratio=1,
            binary_include_bc=True,

            cql=ConservativeSAC.get_default_config(),

            train_type='rl',
            dataset_type='rl',
            save_dataset=False,
            logging=WandBLogger.get_default_config(),
)


def main(argv):

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    set_random_seed(FLAGS.seed)
    
    # load env
    sample_env = gym.make(FLAGS.dense_env).unwrapped
    eval_env = GymnasiumWrapper(gymnasium.make(FLAGS.sparse_env).unwrapped)


 
    if FLAGS.train_type != 'bc':
        behavior_agent = IQLLearner(FLAGS.seed,
                                    sample_env.observation_space.sample()[np.newaxis],
                                    sample_env.action_space.sample()[np.newaxis],
                                    expectile=FLAGS.iql_expectile,
                                    temperature=FLAGS.iql_temperature,
                                    policy_weight_decay=FLAGS.policy_weight_decay,
                                    log_stds=FLAGS.iql_log_stds,
                                    opt_decay_schedule='',
                                    )
    else:
        observation_dim = sample_env.observation_space.shape[0]
        action_dim = sample_env.action_space.shape[0]
        policy = TanhGaussianPolicy(observation_dim, action_dim)
        qf = FullyConnectedQFunction(observation_dim, action_dim)
        behavior_agent = ConservativeSAC(FLAGS.cql, policy, qf)
    
    # load expert policy
    expert_model_pkl_dir = FLAGS.expert_dir
    if expert_model_pkl_dir == '':
        # completely random policy
        intervene_policy = RandomSamplerPolicy(sample_env)
        intervene_agent = None
    elif 'iql' in expert_model_pkl_dir:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_iql_policy_from_model(eval_env, saved_ckpt_expert)
        intervene_agent = saved_ckpt_expert['iql']
        intervene_agent_type = 'iql'
    elif 'rlpd' in expert_model_pkl_dir:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_rlpd_policy_from_model(eval_env, saved_ckpt_expert)
        intervene_agent = saved_ckpt_expert['rlpd']
        intervene_agent_type = 'rlpd'
    else:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_policy_from_model(eval_env, saved_ckpt_expert)
        intervene_agent = saved_ckpt_expert['sac']
        intervene_agent_type = 'sac'
    
    if FLAGS.ground_truth_agent_dir != '':
        if 'iql' in FLAGS.ground_truth_agent_dir:
            ground_truth_agent = load_model(FLAGS.ground_truth_agent_dir)['iql']
            ground_truth_agent_type = 'iql'
        elif 'sac' in FLAGS.ground_truth_agent_dir or 'bc' in FLAGS.ground_truth_agent_dir:
            ground_truth_agent = load_model(FLAGS.ground_truth_agent_dir)['sac']
            ground_truth_agent_type = 'sac'
        elif 'rlpd' in FLAGS.ground_truth_agent_dir:
            ground_truth_agent = load_model(FLAGS.ground_truth_agent_dir)['rlpd']
            ground_truth_agent_type = 'rlpd'
        else:
            raise ValueError("agent type not supported") 
    else:
        ground_truth_agent = FLAGS.ground_truth_agent_dir
        ground_truth_agent_type = ''

    trainer = DAggerBasedLearner(dense_env=FLAGS.dense_env, 
                    sparse_env=FLAGS.sparse_env, 
                    intervene_policy=intervene_policy, 
                    behavior_agent=behavior_agent,
                    dataset_dir=FLAGS.dataset_dir,
                    pretrain_n_epochs=FLAGS.pretrain_n_epochs,
                    pretrain_n_train_step_per_epoch=FLAGS.pretrain_n_train_step_per_epoch,
                    n_epochs=FLAGS.n_epochs,
                    n_train_step_per_epoch=FLAGS.n_train_step_per_epoch,
                    max_traj_length=FLAGS.max_traj_length,
                    collect_n_trajs=FLAGS.collect_n_trajs,
                    batch_size=FLAGS.batch_size,
                    rl_reward_multiplier=FLAGS.rl_reward_multiplier,
                    eval_n_trajs =FLAGS.eval_n_trajs,
                    train_type=FLAGS.train_type,
                    dataset_type=FLAGS.dataset_type,
                    save_dataset=FLAGS.save_dataset,
                    wandb_logger=wandb_logger,
                    intervention_rate=FLAGS.intervention_rate,
                    intervention_strategy=FLAGS.intervention_strategy,
                    ground_truth_agent=ground_truth_agent,
                    intervene_threshold=FLAGS.intervene_threshold,
                    intervene_n_steps=FLAGS.intervene_n_steps,
                    intervene_temperature=FLAGS.intervene_temperature,
                    compare_optimal=FLAGS.compare_optimal,
                    ground_truth_agent_type=ground_truth_agent_type,
                    intervene_agent=intervene_agent,
                    intervene_agent_type=intervene_agent_type,
                    )
    trainer.pretrain()
    results = trainer.run(FLAGS.n_iters)
    print(results)



if __name__ == '__main__':
    absl.app.run(main)