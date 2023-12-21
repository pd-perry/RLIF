import copy
import gym
import d4rl
import numpy as np
from typing import Dict

from gym.spaces import Box
from gym.wrappers.flatten_observation import FlattenObservation
from gymnasium.spaces import Box as gymnasium_box

from ..agents.iql import IQLSamplerPolicy
from ..agents.rlpd import RLPDSamplerPolicy
from ..models.model import SamplerPolicy


def _convert_space(obs_space):
    # breakpoint()
    if isinstance(obs_space, Box) or isinstance(obs_space, gymnasium_box):
        obs_space = Box(obs_space.low, obs_space.high, obs_space.shape)
    elif isinstance(obs_space, gym.spaces.Dict):
        for k, v in obs_space.spaces.items():
            obs_space.spaces[k] = _convert_space(v)
        obs_space = gym.spaces.Dict(obs_space.spaces)
    else:
        raise NotImplementedError
    return obs_space


def _convert_obs(obs):
    if isinstance(obs, np.ndarray):
        if obs.dtype == np.float64:
            return obs.astype(np.float32)
        else:
            return obs
    elif isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = _convert_obs(v)
        return obs


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_space = copy.deepcopy(self.env.observation_space)
        # breakpoint()
        self.observation_space = _convert_space(obs_space)

    def observation(self, observation):
        return _convert_obs(observation)

class UniversalSeed(gym.Wrapper):
    def seed(self, seed: int):
        seeds = self.env.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.action_space.seed(seed)
        return seeds


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    env.action_space = _convert_space(copy.deepcopy(env.action_space))
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    
    return {"return": np.mean(env.return_queue), "normalized_return": env.get_normalized_score(np.sum(list(env.return_queue))), "length": np.mean(env.length_queue)}


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )

def get_d4rl_policy(env):
    dataset = env.get_dataset()
    return

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed

def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)


class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...]
        )


class GymnasiumWrapper():
    def __init__(self, env, train_sparse=False):
        self.env = env
        self.env.obj_range = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.train_sparse = train_sparse
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.train_sparse:
            if reward >= 10:
                reward = 0
            else:
                reward = -1

        return obs, reward, terminated, info
    
    def reset(self):
        observation, _ = self.env.reset()
        self.env.goal = np.array([1.41553579, 0.46682731, 0.42469975])
        return observation
    
    def render(self):
        return self.env.render()

    def transform_dataset(self, dataset):
        mask = dataset['rewards'] >= 10
        dones_shifted = np.append(dataset['dones'][1:], [0])
        dataset['dones'] = dataset['dones'] + dones_shifted
        
        dataset['rewards'] *= mask.astype(int)
        dataset['rewards'] *= dataset['dones']
        dataset['rewards'][dataset['rewards'] > 0] = 1
        return dataset
    

class TrajSampler(object):
    def __init__(self, env, max_traj_length=1000, rlpd=False):
        self.max_traj_length = max_traj_length
        self._env = env
        self.rlpd = rlpd

    def sample(self, policy, n_trajs, replay_buffer=None, deterministic=False, obs_index=False):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            observation = self.env.reset()
            if obs_index:
                    observation = observation['observation']
                
            for _ in range(self.max_traj_length):
                action = policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                next_observation, reward, done, info = self.env.step(action)

                if self.rlpd:
                    if "TimeLimit.truncated" in info:
                        done = True

                if obs_index:
                    next_observation = next_observation['observation']

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )
                
                observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
            ))

        return trajs

    @property
    def env(self):
        return self._env


class UniformSampler(object):
    def __init__(self, env, max_traj_length=1000, intervention_rate=0):
        self.max_traj_length = max_traj_length
        self._env = env
        self.intervention_rate = intervention_rate

        self.iteration = 0

    def sample(self, behavior_policy, intervene_policy, n_trajs, deterministic=False):
        trajs = []
        for _ in range(n_trajs):
            intervene_observations = []
            intervene_actions = []
            intervene_rewards = []
            intervene_next_observations = []
            intervene_dones = []

            first_intervene_action_mask = []
            intervene_tracker = []

            observation = self.env.reset()
            prev_intervene = False
            intervene = False
            next_intervene_time = -1
            sampled_next_time = False
            stop_intervene_time = -1

            for t in range(self.max_traj_length):
                # determine when to intervene
                if not intervene and not sampled_next_time:
                    if self.intervention_rate == 0:
                        next_intervene_time = t + np.random.choice(10) + 1
                    elif self.intervention_rate == 1:
                        next_intervene_time = t + np.random.choice(5) + 1
                    elif self.intervention_rate == 2:
                        next_intervene_time = t + np.random.choice(2) + 1
                    elif self.intervention_rate == 3:
                        next_intervene_time = t + np.random.choice(1) + 1
                    elif self.intervention_rate == 4:
                        next_intervene_time = t

                    sampled_next_time = True
                if t == next_intervene_time:
                    intervene = True
                    if self.intervention_rate <= 0:
                        stop_intervene_time = t + np.random.choice(5) + 1
                    elif self.intervention_rate == 1:
                        stop_intervene_time = t + np.random.choice(5) + 3
                    elif self.intervention_rate == 2:
                        stop_intervene_time = t + np.random.choice(5) + 4
                    elif self.intervention_rate == 3:
                        stop_intervene_time = t + np.random.choice(5) + 12
                    elif self.intervention_rate == 4:
                        stop_intervene_time = 201
                elif t == stop_intervene_time:
                    intervene = False
                    sampled_next_time = False
                
                # sample action
                if intervene:
                    if t != 0 and not prev_intervene:
                        # append state action pair that led to previous intervention
                        first_intervene_action_mask[-1] = 1

                    action = intervene_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                else:
                    action = behavior_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)

                next_observation, reward, done, _ = self.env.step(action)

                # append state and action of intevention trajectory
                intervene_observations.append(observation)
                intervene_actions.append(action)
                intervene_rewards.append(reward)
                intervene_dones.append(done)
                intervene_next_observations.append(next_observation)
                first_intervene_action_mask.append(0)

                observation = next_observation
                prev_intervene = intervene
                intervene_tracker += [intervene]

                if done:
                    break

            trajs.append(dict(
                observations=np.array(intervene_observations, dtype=np.float32),
                actions=np.array(intervene_actions, dtype=np.float32),
                rewards=np.array(intervene_rewards, dtype=np.float32),
                next_observations=np.array(intervene_next_observations, dtype=np.float32),
                dones=np.array(intervene_dones, dtype=np.float32),
                first_intervene_action_mask=np.array(first_intervene_action_mask, dtype=np.float32),
                intervene_tracker=np.array(intervene_tracker, dtype=bool),
            ))

        self.iteration += 1

        return trajs, {}

    @property
    def env(self):
        return self._env


class BinarySampler(object):
    def __init__(self, env, max_traj_length=1000, ground_truth_agent=None, intervene_threshold=10, intervene_n_steps=4, ground_truth_agent_type='sac', compare_optimal=True):
        self.max_traj_length = max_traj_length
        self._env = env
        self.compare_optimal = compare_optimal
        self.ground_truth_agent_type = ground_truth_agent_type

        if self.ground_truth_agent_type == 'sac':
            self.ground_truth_policy = SamplerPolicy(ground_truth_agent.policy, ground_truth_agent.train_params['policy'])
        elif self.ground_truth_agent_type == 'iql':
            self.ground_truth_policy = IQLSamplerPolicy(ground_truth_agent.actor)
        else:
            self.ground_truth_policy = RLPDSamplerPolicy(ground_truth_agent.actor)

        self.epsilon = 5e-2
        self.intervene_threshold = intervene_threshold
        self.intervene_n_steps = intervene_n_steps
        self.iteration = 0
        self.intervene_temperature = 0.1

    def sample(self, behavior_policy, intervene_policy, n_trajs, deterministic=False):
        trajs = []
        for _ in range(n_trajs):
            intervene_observations = []
            intervene_actions = []
            intervene_rewards = []
            intervene_next_observations = []
            intervene_dones = []

            first_intervene_action_mask = []
            intervene_tracker = []

            observation = self.env.reset()
            prev_intervene = False
            intervene = False
            sampled_next_time = False
            stop_intervene_time = -1

            for t in range(self.max_traj_length):
                # determine when to intervene
                expert_action = intervene_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                policy_action = behavior_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                
                optimal_action = self.ground_truth_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                if not intervene:
                    if self.compare_optimal:
                        action_difference = np.linalg.norm((optimal_action - policy_action), ord=1)
                    else:
                        action_difference = np.linalg.norm((expert_action - policy_action), ord=1)

                    epsilon = np.exp(-self.iteration * self.intervene_temperature) * self.epsilon

                    if action_difference < self.intervene_threshold:
                        intervene = np.random.choice([0, 1], p=[1-epsilon, epsilon])
                    else:
                        intervene = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
                        
                    intervene = bool(intervene)

                    if intervene: 
                        stop_intervene_time = t + self.intervene_n_steps

                if t == stop_intervene_time:
                    intervene = False
                    sampled_next_time = False

                # sample action
                if intervene:
                    if t != 0 and not prev_intervene:
                        # append state action pair that led to previous intervention
                        first_intervene_action_mask[-1] = 1

                    action = expert_action
                else:
                    action = policy_action

                next_observation, reward, done, _ = self.env.step(action)

                # append state and action of intevention trajectory
                intervene_observations.append(observation)
                intervene_actions.append(action)
                intervene_rewards.append(reward)
                intervene_dones.append(done)
                intervene_next_observations.append(next_observation)
                first_intervene_action_mask.append(0)

                observation = next_observation
                prev_intervene = intervene
                intervene_tracker += [intervene]

                if done:
                    break
            trajs.append(dict(
                observations=np.array(intervene_observations, dtype=np.float32),
                actions=np.array(intervene_actions, dtype=np.float32),
                rewards=np.array(intervene_rewards, dtype=np.float32),
                next_observations=np.array(intervene_next_observations, dtype=np.float32),
                dones=np.array(intervene_dones, dtype=np.float32),
                first_intervene_action_mask=np.array(first_intervene_action_mask, dtype=np.float32),
                intervene_tracker=np.array(intervene_tracker, dtype=bool),
            ))
            self.iteration += 1

        return trajs, {}

    @property
    def env(self):
        return self._env


class DaggerBinarySampler(object):
    def __init__(self, env, max_traj_length=1000, ground_truth_agent=None, intervene_threshold=0, ground_truth_agent_type='sac'):
        self.max_traj_length = max_traj_length
        self._env = env
        self.ground_truth_agent = ground_truth_agent
        self.ground_truth_agent_type = ground_truth_agent_type

        if self.ground_truth_agent_type == 'sac':
            self.ground_truth_policy = SamplerPolicy(self.ground_truth_agent.policy, self.ground_truth_agent.train_params['policy'])
        elif self.ground_truth_agent_type == 'iql':
            self.ground_truth_policy = IQLSamplerPolicy(ground_truth_agent.actor)
        else:
            self.ground_truth_policy = RLPDSamplerPolicy(ground_truth_agent.actor)

        self.epsilon = 5e-2
        self.rate = 0
        self.intervene_threshold = intervene_threshold

        self.intervene_temperature = 0.1
        self.iteration = 0

    def sample(self, behavior_policy, intervene_policy, n_trajs, deterministic=False):
        trajs = []
        for _ in range(n_trajs):
            intervene_observations = []
            intervene_actions = []
            intervene_rewards = []
            intervene_next_observations = []
            intervene_dones = []
            expert_actions = []

            first_intervene_action_mask = []
            intervene_tracker = []

            observation = self.env.reset()
            intervene = False
            sampled_next_time = False

            for t in range(self.max_traj_length):
                expert_action = intervene_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                expert_actions += [expert_action]
                policy_action = behavior_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                
                action = policy_action

                next_observation, reward, done, _ = self.env.step(action)

                # append state and action of intevention trajectory
                intervene_observations.append(observation)
                intervene_actions.append(action)
                intervene_rewards.append(reward)
                intervene_dones.append(done)
                intervene_next_observations.append(next_observation)
                first_intervene_action_mask.append(0)

                observation = next_observation
                intervene_tracker += [intervene]

                if done:
                    break
            for i in range(len(intervene_actions)):
                action_difference = np.linalg.norm((expert_actions[i] - intervene_actions[i]), ord=1)
                epsilon = 0.05
                if action_difference < self.intervene_threshold:
                    intervene = np.random.choice([0, 1], p=[1-epsilon, epsilon])
                else:
                    intervene = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
                    
                intervene = bool(intervene)

                if intervene:
                    intervene_actions[i] = expert_actions[i]
                    intervene_tracker[i] = True
            
            trajs.append(dict(
                observations=np.array(intervene_observations, dtype=np.float32),
                actions=np.array(intervene_actions, dtype=np.float32),
                rewards=np.array(intervene_rewards, dtype=np.float32),
                next_observations=np.array(intervene_next_observations, dtype=np.float32),
                dones=np.array(intervene_dones, dtype=np.float32),
                first_intervene_action_mask=np.array(first_intervene_action_mask, dtype=np.float32),
                intervene_tracker=np.array(intervene_tracker, dtype=bool),
            ))
            self.iteration += 1

        return trajs, {}

    @property
    def env(self):
        return self._env
    

class DaggerUniformSampler(object):
    def __init__(self, env, max_traj_length=1000, ground_truth_agent=None, intervention_rate=0, ground_truth_agent_type='sac'):
        self.max_traj_length = max_traj_length
        self._env = env
        self.ground_truth_agent = ground_truth_agent
        self.ground_truth_agent_type = ground_truth_agent_type

        if self.ground_truth_agent_type == 'sac':
            self.ground_truth_policy = SamplerPolicy(self.ground_truth_agent.policy, self.ground_truth_agent.train_params['policy'])
        elif self.ground_truth_agent_type == 'iql':
            self.ground_truth_policy = IQLSamplerPolicy(ground_truth_agent.actor)
        else:
            self.ground_truth_policy = RLPDSamplerPolicy(ground_truth_agent.actor)

        self.epsilon = 5e-2
        self.rate = 0
        if intervention_rate == 0:
            self.rate = 0.3
        elif intervention_rate == 1:
            self.rate = 0.5
        elif intervention_rate == 3:
            self.rate = 0.875

        self.intervene_temperature = 0.1
        self.iteration = 0

    def sample(self, behavior_policy, intervene_policy, n_trajs, deterministic=False):
        trajs = []
        for _ in range(n_trajs):
            intervene_observations = []
            intervene_actions = []
            intervene_rewards = []
            intervene_next_observations = []
            intervene_dones = []

            first_intervene_action_mask = []
            intervene_tracker = []

            observation = self.env.reset()
            intervene = False
            sampled_next_time = False

            for t in range(self.max_traj_length):
                action = behavior_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)

                next_observation, reward, done, _ = self.env.step(action)

                # append state and action of intevention trajectory
                intervene_observations.append(observation)
                intervene_actions.append(action)
                intervene_rewards.append(reward)
                intervene_dones.append(done)
                intervene_next_observations.append(next_observation)
                first_intervene_action_mask.append(0)

                observation = next_observation
                intervene_tracker += [intervene]

                if done:
                    break
            
            relabel_index = np.random.choice(list(range(len(intervene_observations))), int(len(intervene_observations) * self.rate))
            for i in relabel_index:
                intervene_actions[i] = intervene_policy(intervene_observations[i].reshape(1, -1), deterministic=deterministic).reshape(-1)
                intervene_tracker[i] = True
            
            trajs.append(dict(
                observations=np.array(intervene_observations, dtype=np.float32),
                actions=np.array(intervene_actions, dtype=np.float32),
                rewards=np.array(intervene_rewards, dtype=np.float32),
                next_observations=np.array(intervene_next_observations, dtype=np.float32),
                dones=np.array(intervene_dones, dtype=np.float32),
                first_intervene_action_mask=np.array(first_intervene_action_mask, dtype=np.float32),
                intervene_tracker=np.array(intervene_tracker, dtype=bool),
            ))
            self.iteration += 1

        return trajs, {}

    @property
    def env(self):
        return self._env


class ThresholdSampler(object):
    def __init__(self, env, max_traj_length=1000, ground_truth_agent=None, intervene_threshold=10, intervene_temperature=0, intervene_n_steps=5, ground_truth_agent_type='sac'):
        self.max_traj_length = max_traj_length
        self._env = env
        self.ground_truth_agent = ground_truth_agent
        self.ground_truth_agent_type = ground_truth_agent_type

        if self.ground_truth_agent_type == 'sac':
            self.ground_truth_policy = SamplerPolicy(self.ground_truth_agent.policy, self.ground_truth_agent.train_params['policy'])
        elif self.ground_truth_agent_type == 'iql':
            self.ground_truth_policy = IQLSamplerPolicy(ground_truth_agent.actor)
        else:
            self.ground_truth_policy = RLPDSamplerPolicy(ground_truth_agent.actor)

        self.epsilon = 5e-2
        self.intervene_threshold = intervene_threshold
        self.intervene_temperature = 0.1
        self.iteration = 0
        self.intervene_n_steps = intervene_n_steps

    def sample(self, behavior_policy, intervene_policy, n_trajs, deterministic=False):
        trajs = []
        for _ in range(n_trajs):
            intervene_observations = []
            intervene_actions = []
            intervene_rewards = []
            intervene_next_observations = []
            intervene_dones = []

            first_intervene_action_mask = []
            intervene_tracker = []

            observation = self.env.reset()
            prev_intervene = False
            intervene = False
            sampled_next_time = False
            stop_intervene_time = -1

            for t in range(self.max_traj_length):
                expert_action = intervene_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                policy_action = behavior_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                ground_truth_action = self.ground_truth_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)

                if not intervene:
                    if self.ground_truth_agent_type == 'iql':
                        gt_q1, gt_q2 = self.ground_truth_agent.critic(observation, ground_truth_action)
                        gt_q = np.min([gt_q1, gt_q2])
                        policy_q1, policy_q2 = self.ground_truth_agent.critic(observation, policy_action)

                        policy_q = np.min([policy_q1, policy_q2])
                    elif self.ground_truth_agent_type == 'sac':
                        gt_q1 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf1'], observation, ground_truth_action)
                        gt_q2 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf2'], observation, ground_truth_action)
                        gt_q = np.min([gt_q1, gt_q2])

                        policy_q1 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf1'], observation, policy_action)
                        policy_q2 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf2'], observation, policy_action)
                        policy_q = np.min([policy_q1, policy_q2])
                    else:
                        gt_qs = self.ground_truth_agent.critic.apply_fn(
                            {"params": self.ground_truth_agent.critic.params},
                            observation,
                            ground_truth_action,
                            True,
                        )  # training=True
                        gt_q = gt_qs.mean(axis=0)

                        policy_qs = self.ground_truth_agent.critic.apply_fn(
                            {"params": self.ground_truth_agent.critic.params},
                            observation,
                            policy_action,
                            True,
                        )  # training=True
                        policy_q = policy_qs.mean(axis=0)

                    epsilon = np.exp(-self.iteration * self.intervene_temperature) * self.epsilon

                    if policy_q < gt_q * self.intervene_threshold:
                        intervene = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
                    else:
                        intervene = np.random.choice([0, 1], p=[1-epsilon, epsilon])
                        
                    intervene = bool(intervene)

                    if intervene: 
                        stop_intervene_time = t + self.intervene_n_steps

                if t == stop_intervene_time:
                    intervene = False
                    sampled_next_time = False
                
                # sample action
                if intervene:
                    if t != 0 and not prev_intervene:
                        # append state action pair that led to previous intervention
                        first_intervene_action_mask[-1] = 1

                    action = expert_action
                else:
                    action = policy_action

                next_observation, reward, done, _ = self.env.step(action)

                # append state and action of intevention trajectory
                intervene_observations.append(observation)
                intervene_actions.append(action)
                intervene_rewards.append(reward)
                intervene_dones.append(done)
                intervene_next_observations.append(next_observation)
                first_intervene_action_mask.append(0)

                observation = next_observation
                prev_intervene = intervene
                intervene_tracker += [intervene]

                if done:
                    break
            trajs.append(dict(
                observations=np.array(intervene_observations, dtype=np.float32),
                actions=np.array(intervene_actions, dtype=np.float32),
                rewards=np.array(intervene_rewards, dtype=np.float32),
                next_observations=np.array(intervene_next_observations, dtype=np.float32),
                dones=np.array(intervene_dones, dtype=np.float32),
                first_intervene_action_mask=np.array(first_intervene_action_mask, dtype=np.float32),
                intervene_tracker=np.array(intervene_tracker, dtype=bool),
            ))
            self.iteration += 1

        return trajs, {}

    @property
    def env(self):
        return self._env


class ThresholdLabelSampler(object):
    def __init__(self, env, max_traj_length=1000, ground_truth_agent=None, intervene_threshold=10, intervene_temperature=0, intervene_n_steps=5, ground_truth_agent_type='sac'):
        self.max_traj_length = max_traj_length
        self._env = env
        self.ground_truth_agent = ground_truth_agent
        self.ground_truth_agent_type = ground_truth_agent_type

        if self.ground_truth_agent_type == 'sac':
            self.ground_truth_policy = SamplerPolicy(self.ground_truth_agent.policy, self.ground_truth_agent.train_params['policy'])
        elif self.ground_truth_agent_type == 'iql':
            self.ground_truth_policy = IQLSamplerPolicy(ground_truth_agent.actor)
        else:
            self.ground_truth_policy = RLPDSamplerPolicy(ground_truth_agent.actor)

        self.epsilon = 5e-2
        self.intervene_threshold = intervene_threshold
        self.intervene_temperature = 0.1
        self.iteration = 0
        self.intervene_n_steps = intervene_n_steps

    def sample(self, behavior_policy, intervene_policy, n_trajs, deterministic=False):
        trajs = []
        for _ in range(n_trajs):
            intervene_observations = []
            intervene_actions = []
            intervene_rewards = []
            intervene_next_observations = []
            intervene_dones = []

            first_intervene_action_mask = []
            intervene_tracker = []

            observation = self.env.reset()
            prev_intervene = False
            intervene = False
            sampled_next_time = False
            stop_intervene_time = -1

            for t in range(self.max_traj_length):
                # determine when to intervene
                expert_action = intervene_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                policy_action = behavior_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)
                ground_truth_action = self.ground_truth_policy(observation.reshape(1, -1), deterministic=deterministic).reshape(-1)

                if not intervene:
                    if self.ground_truth_agent_type == 'iql':
                        gt_q1, gt_q2 = self.ground_truth_agent.critic(observation, ground_truth_action)
                        gt_q = np.min([gt_q1, gt_q2])
                        policy_q1, policy_q2 = self.ground_truth_agent.critic(observation, policy_action)

                        policy_q = np.min([policy_q1, policy_q2])
                    elif self.ground_truth_agent_type == 'sac':
                        gt_q1 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf1'], observation, ground_truth_action)
                        gt_q2 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf2'], observation, ground_truth_action)
                        gt_q = np.min([gt_q1, gt_q2])

                        policy_q1 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf1'], observation, policy_action)
                        policy_q2 = self.ground_truth_agent.qf.apply(self.ground_truth_agent.train_params['qf2'], observation, policy_action)
                        policy_q = np.min([policy_q1, policy_q2])
                    else:
                        gt_qs = self.ground_truth_agent.critic.apply_fn(
                            {"params": self.ground_truth_agent.critic.params},
                            observation,
                            ground_truth_action,
                            True,
                        )  # training=True
                        gt_q = gt_qs.mean(axis=0)

                        policy_qs = self.ground_truth_agent.critic.apply_fn(
                            {"params": self.ground_truth_agent.critic.params},
                            observation,
                            policy_action,
                            True,
                        )  # training=True
                        policy_q = policy_qs.mean(axis=0)

                    epsilon = np.exp(-self.iteration * self.intervene_temperature) * self.epsilon

                    if policy_q < gt_q * self.intervene_threshold:
                        intervene = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
                    else:
                        intervene = np.random.choice([0, 1], p=[1-epsilon, epsilon])
                        
                    intervene = bool(intervene)

                    if intervene: 
                        stop_intervene_time = t + self.intervene_n_steps

                if t == stop_intervene_time:
                    intervene = False
                    sampled_next_time = False
                
                # sample action
                if intervene:
                    if t != 0 and not prev_intervene:
                        # append state action pair that led to previous intervention
                        first_intervene_action_mask[-1] = 1

                    action = policy_action
                else:
                    action = policy_action

                next_observation, reward, done, _ = self.env.step(action)

                # append state and action of intevention trajectory
                intervene_observations.append(observation)
                intervene_actions.append(action)
                intervene_rewards.append(reward)
                intervene_dones.append(done)
                intervene_next_observations.append(next_observation)
                first_intervene_action_mask.append(0)

                observation = next_observation
                prev_intervene = intervene
                intervene_tracker += [intervene]

                if done:
                    break
            trajs.append(dict(
                observations=np.array(intervene_observations, dtype=np.float32),
                actions=np.array(intervene_actions, dtype=np.float32),
                rewards=np.array(intervene_rewards, dtype=np.float32),
                next_observations=np.array(intervene_next_observations, dtype=np.float32),
                dones=np.array(intervene_dones, dtype=np.float32),
                first_intervene_action_mask=np.array(first_intervene_action_mask, dtype=np.float32),
                intervene_tracker=np.array(intervene_tracker, dtype=bool),
            ))
            self.iteration += 1

        return trajs, {}

    @property
    def env(self):
        return self._env
