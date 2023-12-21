import copy
from functools import partial
import cloudpickle as pickle

import numpy as np
import wandb
import distrax
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import tqdm

import gym
import gymnasium
import d4rl

from ..utils.env_utils import (
    DaggerBinarySampler, DaggerUniformSampler, TrajSampler, UniformSampler, BinarySampler,
    ThresholdSampler, ThresholdLabelSampler, GymnasiumWrapper, get_d4rl_dataset, subsample_batch
)
from ..models.model import SamplerPolicy, evaluate_policy
from .iql import Batch, IQLSamplerPolicy
from ..utils.dataset_utils import OfflineDataset
from ..utils.jax_utils import batch_to_jax
from ..utils.utils import (
    Timer, prefix_metrics
)


class DAggerBasedLearner(object):
    """Supports DAgger, HG-DAgger, and IQL HG-DAgger Baselines."""

    def __init__(self, 
                 dense_env, 
                 sparse_env, 
                 intervene_policy, 
                 behavior_agent,
                 dataset_dir,
                 pretrain_n_epochs,
                 pretrain_n_train_step_per_epoch,
                 n_epochs, 
                 n_train_step_per_epoch, 
                 max_traj_length, 
                 collect_n_trajs,
                 batch_size,
                 rl_reward_multiplier,
                 eval_n_trajs,
                 train_type,
                 dataset_type,
                 save_dataset,
                 wandb_logger,
                 intervention_rate,
                 intervention_strategy,
                 ground_truth_agent,
                 intervene_threshold,
                 intervene_n_steps,
                 intervene_temperature,
                 compare_optimal, 
                 ground_truth_agent_type,
                 intervene_agent,
                 intervene_agent_type):
        self.env_name = dense_env
        self.train_env = gym.make(dense_env).unwrapped
        self.eval_env = GymnasiumWrapper(gymnasium.make(sparse_env).unwrapped)
        self.intervene_policy = intervene_policy
        self.behavior_agent = behavior_agent
        self.intervene_agent = intervene_agent
        self.intervene_agent_type = intervene_agent_type

        self.wandb_logger = wandb_logger
        self.dataset_dir = dataset_dir
        
        self.rl_reward_multiplier = rl_reward_multiplier
        self.batch_size = batch_size
        self.pretrain_n_epochs = pretrain_n_epochs
        self.pretrain_n_train_step_per_epoch = pretrain_n_train_step_per_epoch
        self.n_epochs = n_epochs
        self.n_train_step_per_epoch = n_train_step_per_epoch
        self.max_traj_length = max_traj_length
        self.collect_n_trajs = collect_n_trajs
        self.eval_n_trajs = eval_n_trajs

        self.intervention_rate = intervention_rate
        self.intervention_strategy = intervention_strategy
        self.ground_truth_agent = ground_truth_agent
        self.intervene_threshold = intervene_threshold
        self.intervene_temperature = intervene_temperature
        self.intervene_n_steps =intervene_n_steps
        self.compare_optimal = compare_optimal
        self.ground_truth_agent_type = ground_truth_agent_type

        self.setup_data()
        self.setup_sampler()
        
        self.success_function = lambda t: np.all(t['rewards'][-1]>=10)
        self.train_type = train_type
        self.dataset_type = dataset_type
        self.save_dataset = save_dataset

    def setup_data(self):
        if self.dataset_dir != '':
            with open(self.dataset_dir, 'rb') as handle:
                self.dataset = pickle.load(handle)
        else:
            self.dataset = get_d4rl_dataset(self.train_env)

        self.dataset['first_intervene_action_mask'] = np.zeros_like(self.dataset['rewards'])
        self.dataset['intervene_tracker'] = np.zeros_like(self.dataset['rewards'])
        self.dataset['actions'] = np.clip(self.dataset['actions'], -0.999, 0.999)
        self.dataset['rewards'] = np.zeros_like(self.dataset['rewards'])

        self.rl_dataset = copy.deepcopy(self.dataset)
    
    def setup_sampler(self):
        self.sparse_eval_sampler = TrajSampler(self.eval_env, self.max_traj_length)
        self.dense_eval_sampler = TrajSampler(self.train_env, self.max_traj_length)
        if self.intervention_strategy == 'dagger':
            self.intervene_sampler = DaggerUniformSampler(self.train_env, self.max_traj_length, self.ground_truth_agent, self.intervention_rate, ground_truth_agent_type=self.ground_truth_agent_type)
        elif self.intervention_strategy == 'dagger_binary':
            self.intervene_sampler = DaggerBinarySampler(self.train_env, self.max_traj_length, self.ground_truth_agent, self.intervene_threshold, ground_truth_agent_type=self.ground_truth_agent_type)
        elif self.intervention_strategy == 'binary':
            self.intervene_sampler = BinarySampler(self.train_env, self.max_traj_length, self.ground_truth_agent, self.intervene_threshold, self.intervene_n_steps, self.ground_truth_agent_type, self.compare_optimal)
        elif self.intervention_strategy == 'threshold':
            self.intervene_sampler = ThresholdSampler(self.train_env, self.max_traj_length, self.ground_truth_agent, self.intervene_threshold, self.intervene_temperature, self.intervene_n_steps, self.ground_truth_agent_type)
        elif self.intervention_strategy == 'threshold_label':
            self.intervene_sampler = ThresholdLabelSampler(self.train_env, self.max_traj_length, self.ground_truth_agent, self.intervene_threshold, self.intervene_temperature, self.intervene_n_steps, self.ground_truth_agent_type)
        else:
            self.intervene_sampler = UniformSampler(self.train_env, self.max_traj_length, self.intervention_rate)

    def pretrain(self):
        if self.train_type == 'bc':
            dataset = self.dataset
            for i in tqdm.tqdm(range(0, self.pretrain_n_epochs),
                                smoothing=0.1,
                                disable=False):
                for batch_idx in range(self.pretrain_n_train_step_per_epoch):
                    batch = batch_to_jax(subsample_batch(dataset, self.batch_size))
                    self.behavior_agent.train(batch, bc=True)
            
        else:
            if self.dataset_type == "bc":
                dataset = OfflineDataset(self.dataset)
            else:
                dataset = OfflineDataset(self.rl_dataset)

            for i in tqdm.tqdm(range(0, self.pretrain_n_epochs),
                        smoothing=0.1,
                        disable=False):
                for batch_idx in range(self.pretrain_n_train_step_per_epoch):
                    batch = dataset.sample(self.batch_size)
                    update_info = self.behavior_agent.update(batch)
        print("PRETRAINING DONE")

    def run(self, n_iters):
        success_rates = []
        average_success_rate = self.evaluate()
        print("initial success rate: ", average_success_rate)
        success_rates += [average_success_rate]

        for i in range(n_iters):
            if self.train_type != 'bc':
                behavior_policy = IQLSamplerPolicy(self.behavior_agent.actor)
            else:
                behavior_policy = SamplerPolicy(self.behavior_agent.policy, self.behavior_agent.train_params['policy'])

            intervene_dataset, metrics = self.intervene_sampler.sample(behavior_policy=behavior_policy,
                                                    intervene_policy=self.intervene_policy,
                                                    n_trajs=self.collect_n_trajs
                                                    )
            self.dataset, intervene_tracker = self.concat_dataset(intervene_dataset, self.dataset)
            self.rl_dataset = copy.deepcopy(self.dataset)
            self.rl_dataset['rewards'] = -1 * self.rl_reward_multiplier * self.rl_dataset['first_intervene_action_mask']

            # perform training with intervention dataset
            average_success_rate = self.train_and_eval(iter_num=i+1, intervene_tracker=intervene_tracker, q_diff=metrics)
            print(f"success rates iter {i+1}: ", average_success_rate)
            success_rates += [average_success_rate]
        
        if self.save_dataset:
            if self.dataset_type == 'rl':
                with open(f'./intervene/datasets/{self.dataset_type}_env_{self.env_name}_n_iters_{n_iters}_n_trajs_{self.collect_n_trajs}_n_epochs_{self.n_epochs}_scale_{self.rl_reward_multiplier}', "wb") as f:
                    pickle.dump(self.rl_dataset, f)
            elif self.dataset_type == 'bc':
                with open(f'./intervene/datasets/{self.dataset_type}_env_{self.env_name}_n_iters_{n_iters}_n_trajs_{self.collect_n_trajs}_n_epochs_{self.n_epochs}', "wb") as f:
                    pickle.dump(self.dataset, f)

        return success_rates

    def combine_batch(self, batch1, batch2):
        observations = np.vstack([batch1.observations, batch2.observations])
        actions = np.vstack([batch1.actions, batch2.actions])
        rewards = np.hstack([batch1.rewards, batch2.rewards])
        masks = np.hstack([batch1.masks, batch2.masks])
        next_observations = np.vstack([batch1.next_observations, batch2.next_observations])
        
        return Batch(observations=observations,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     next_observations=next_observations)

    def train_and_eval(self, iter_num, intervene_tracker, q_diff):
        if self.train_type == 'bc':
            dataset = self.dataset
        else:
            if self.dataset_type == "bc":
                dataset = OfflineDataset(self.dataset)
            else:
                dataset = OfflineDataset(self.rl_dataset)
        
        metrics = {'steps': iter_num}

        with Timer() as train_timer:
            for epoch in range(self.n_epochs):
                metrics = {'epoch': epoch}
                
                if self.train_type != 'bc':
                    for batch_idx in range(self.n_train_step_per_epoch):
                        batch = dataset.sample(self.batch_size)
                        update_info = self.behavior_agent.update(batch)
                        update_info['adv'] = np.mean(update_info['adv'])
                        metrics.update(prefix_metrics(update_info, 'iql'))
                        metrics.update(prefix_metrics(q_diff, 'iql'))
                else:
                    for batch_idx in range(self.n_train_step_per_epoch):
                        batch = batch_to_jax(subsample_batch(dataset, self.batch_size))
                        metrics.update(prefix_metrics(self.behavior_agent.train(batch, bc=True), 'sac'))
        
        with Timer() as eval_timer:
            if self.train_type != 'bc':
                sampler_policy = IQLSamplerPolicy(self.behavior_agent.actor)
            else:
                sampler_policy = SamplerPolicy(self.behavior_agent.policy, self.behavior_agent.train_params['policy'])

            sparse_trajs = self.sparse_eval_sampler.sample(
                        sampler_policy,
                        self.eval_n_trajs, deterministic=False
                    )
            dense_trajs = self.dense_eval_sampler.sample(
                sampler_policy,
                self.eval_n_trajs, deterministic=False
            )
            
            success_rate = evaluate_policy(sparse_trajs, 
                                        success_rate=True,
                                        success_function=self.success_function)

            reward_50_ct = np.mean([np.sum(t['rewards'] > 50) for t in dense_trajs])

            metrics['reward_bonus_ct'] =  reward_50_ct
            metrics['average_success_rate'] = success_rate 
            metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in dense_trajs])
            metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in dense_trajs])
            metrics['average_normalizd_return'] = np.mean(
                [self.dense_eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in dense_trajs]
            )
            metrics['intervene_rate'] = np.mean(intervene_tracker)

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        self.wandb_logger.log(metrics)
        return success_rate

    def evaluate(self):
        metrics = {'steps': 0}
        with Timer() as eval_timer:
            if self.train_type != 'bc':
                sampler_policy = IQLSamplerPolicy(self.behavior_agent.actor)
            else:
                sampler_policy = SamplerPolicy(self.behavior_agent.policy, self.behavior_agent.train_params['policy'])
        
            sparse_trajs = self.sparse_eval_sampler.sample(
                        sampler_policy,
                        self.eval_n_trajs, deterministic=False
                    )
            dense_trajs = self.dense_eval_sampler.sample(
                sampler_policy,
                self.eval_n_trajs, deterministic=False
            )
            
            success_rate = evaluate_policy(sparse_trajs, 
                                        success_rate=True,
                                        success_function=self.success_function)

            reward_50_ct = np.mean([np.sum(t['rewards'] > 50) for t in dense_trajs])

            metrics['reward_bonus_ct'] =  reward_50_ct
            metrics['average_success_rate'] = success_rate 
            metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in dense_trajs])
            metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in dense_trajs])
            metrics['average_normalizd_return'] = np.mean(
                [self.dense_eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in dense_trajs]
            )

        self.wandb_logger.log(metrics)
        return success_rate
    
    def concat_dataset(self, dataset, base_dataset, rl=False):
        observations = np.vstack([dataset[i]['observations'] for i in range(len(dataset))])
        actions = np.vstack([dataset[i]['actions'] for i in range(len(dataset))])
        next_observations = np.vstack([dataset[i]['next_observations'] for i in range(len(dataset))])

        dones = np.hstack([dataset[i]['dones'] for i in range(len(dataset))])
        first_intervene_action_mask = np.hstack([dataset[i]['first_intervene_action_mask'] for i in range(len(dataset))])
        intervene_tracker = np.hstack([dataset[i]['intervene_tracker'] for i in range(len(dataset))])
        rewards = np.full(dones.shape, 0)

        base_dataset['observations'] = np.vstack([base_dataset['observations'], observations])
        base_dataset['actions'] = np.vstack([base_dataset['actions'], actions])
        base_dataset['rewards'] = np.hstack([base_dataset['rewards'], rewards])
        base_dataset['next_observations'] = np.vstack([base_dataset['next_observations'], next_observations])
        base_dataset['dones'] = np.hstack([base_dataset['dones'], dones])
        base_dataset['first_intervene_action_mask'] = np.hstack([base_dataset['first_intervene_action_mask'], first_intervene_action_mask])
        base_dataset['intervene_tracker'] = np.hstack([base_dataset['intervene_tracker'], intervene_tracker])
        base_dataset['actions'] = np.clip(base_dataset['actions'], -0.999, 0.999)
        
        return base_dataset, intervene_tracker


     