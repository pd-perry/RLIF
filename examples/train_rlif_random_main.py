import os
import pickle

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import gym
import numpy as np
import tqdm
from absl import app, flags
import cloudpickle as pickle

import gymnasium

from ..utils.env_utils import GymnasiumWrapper, TrajSampler, get_d4rl_dataset, evaluate, wrap_gym

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from ..agents.rlpd import RLPDSamplerPolicy, get_rlpd_policy_from_model, SACLearner
from ..agents.iql import get_iql_policy_from_model, IQLSamplerPolicy
from ..utils.dataset_utils import ReplayBuffer
from ..utils.dataset_utils import D4RLDataset

from ..models.model import SamplerPolicy, get_policy_from_model, load_model, evaluate_policy

from ..utils.utils import define_flags_with_default, set_random_seed


FLAGS_DEF = define_flags_with_default(
    project_name="rlpd_itv_test",
    env_name="pen-expert-v1",
    sparse_env='AdroitHandPenSparse-v1',
    offline_ratio=0.5,
    seed=43,
    train_sparse=False,
    dataset_dir='',

    expert_dir='./intervene/bc_output/grace_bc_adroit_d4rl_policies/994fb9e340c94a2aa94a9685bb64b3ae/model.pkl',
    ground_truth_agent_dir='./intervene/grace_rlpd_expert_adroit/s24_0pretrain_15utd_0.3offline_LN/model/model.pkl',
    intervene_threshold=0.0,
    intervention_strategy='',
    intervention_rate=1,
    intervene_n_steps=4,

    eval_episodes=100,
    log_interval=1000,
    eval_interval=10000,
    max_traj_length=200,
    batch_size=256,
    max_steps=int(1e6),
    start_training=0,
    pretrain_steps=0,

    tqdm=True,
    save_video=False,
    save_model=False,
    checkpoint_model=False,
    checkpoint_buffer=False,
    utd_ratio=1,
    binary_include_bc=True,
    )



config_flags.DEFINE_config_file(
    "config",
    "./RLIF/configs/rlpd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def combine(one_dict, other_dict):
    combined = {}
    for k, v in one_dict.items():
        if len(v.shape) > 1:
            tmp = np.vstack((v, other_dict[k]))
        else:
            tmp = np.hstack((v, other_dict[k]))
        combined[k] = tmp
    return combined


def main(_):
    FLAGS = flags.FLAGS
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    set_random_seed(FLAGS.seed)

    wandb.init(project=FLAGS.project_name, mode='online')
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain_{FLAGS.utd_ratio}utd_{FLAGS.offline_ratio}offline"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)
    
    if FLAGS.save_model:
        model_dir = os.path.join(log_dir, "model")
        os.makedirs(model_dir, exist_ok=True)


    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    ds = D4RLDataset(env)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    sparse_eval_sampler = TrajSampler(GymnasiumWrapper(gymnasium.make(FLAGS.sparse_env).unwrapped), FLAGS.max_traj_length)

    # load agents
    expert_model_pkl_dir = FLAGS.expert_dir
    if 'iql' in expert_model_pkl_dir:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_iql_policy_from_model(eval_env, saved_ckpt_expert)
    elif 'rlpd' in expert_model_pkl_dir:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_rlpd_policy_from_model(eval_env, saved_ckpt_expert)
    else:
        saved_ckpt_expert = load_model(expert_model_pkl_dir)
        intervene_policy = get_policy_from_model(eval_env, saved_ckpt_expert)


    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    if FLAGS.dataset_dir != '':
            with open(FLAGS.dataset_dir, 'rb') as handle:
                dataset = pickle.load(handle)
    else:
        dataset = get_d4rl_dataset(env)

    dataset['actions'] = np.clip(dataset['actions'], -0.999, 0.999)
    dataset['rewards'] = np.zeros_like(dataset['rewards'])
    dataset['masks'] = 1 - dataset['dones']

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    for i in range(len(dataset['rewards'])):
        replay_buffer.insert(
            dict(
                observations=dataset['observations'][i],
                actions=dataset['actions'][i],
                rewards=0,
                masks=dataset['masks'][i],
                dones=dataset['dones'][i],
                next_observations=dataset['next_observations'][i],
            )
        )
    

    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)

            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=i)

            sampler_policy = RLPDSamplerPolicy(agent.actor)
            sparse_trajs = sparse_eval_sampler.sample(
                    sampler_policy,
                    FLAGS.eval_episodes, deterministic=False
                )
            avg_success = evaluate_policy(sparse_trajs,
                                            success_rate=True,
                                            success_function=lambda t: np.all(t['rewards'][-1:]>=10),
                                            )
            wandb.log({f"offline-evaluation/avg_success": avg_success}, step=i)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_masks = []
    all_dones = []
    all_next_observations = []
    all_intervene = []


    observation, done = env.reset(), False
    t = 0
    intervene = False
    prev_intervene = False
    stop_intervene_time = -1
    first_intervene_action_mask = []
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        
        policy_action, agent = agent.sample_actions(observation)
        expert_action = intervene_policy(observation.reshape(1, -1), deterministic=False).reshape(-1)

        intervene = False
        if not intervene:
            if FLAGS.intervention_rate == 0:
                # regular uniform intervene
                intervene = bool(np.random.choice(a=[0, 1], size=1, p=[0.7, 0.3]))
            elif FLAGS.intervention_rate == 1:
                intervene = bool(np.random.choice(a=[0, 1], size=1, p=[0.5, 0.5]))
            elif FLAGS.intervention_rate == 2:
                intervene = bool(np.random.choice(a=[0, 1], size=1, p=[0.3, 0.7]))
            elif FLAGS.intervention_rate == 3:
                intervene = bool(np.random.choice(a=[0, 1], size=1, p=[0.15, 0.85]))
            elif FLAGS.intervention_rate == 4:
                intervene = True

            if intervene: 
                stop_intervene_time = t + FLAGS.intervene_n_steps

        if t == stop_intervene_time:
            intervene = False
        

        if intervene:
            if t != 0 and not prev_intervene:
                # append state action pair that led to previous intervention
                first_intervene_action_mask[-1] = 1
            
                replay_buffer.insert(
                   dict(
                        observations=all_observations[-1],
                        actions=all_actions[-1],
                        rewards=-1,
                        masks=all_masks[-1],
                        dones=all_dones[-1],
                        next_observations=all_next_observations[-1],
                    )
                )
            if 'label' in FLAGS.intervention_strategy:
                action = policy_action
            else:
                action = expert_action
        else:
            action = policy_action

            if t != 0:
                replay_buffer.insert(
                   dict(
                        observations=all_observations[-1],
                        actions=all_actions[-1],
                        rewards=0,
                        masks=all_masks[-1],
                        dones=all_dones[-1],
                        next_observations=all_next_observations[-1],
                    )
                )

        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        prev_intervene = intervene
        all_observations += [observation]
        all_actions += [action]
        all_rewards += [0]
        all_masks += [mask]
        all_dones += [done]
        all_next_observations += [next_observation]
        first_intervene_action_mask.append(0)
        all_intervene += [intervene]
        t += 1

        observation = next_observation

        if done or t > FLAGS.max_traj_length:
            observation, done = env.reset(), False
            intervene = False
            prev_intervene = False
            stop_intervene_time = -1
            t = 0
            try:
                for k, v in info["episode"].items():
                    decode = {"r": "return", "l": "length", "t": "time"}
                    wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)
            except:
                pass

        online_batch = replay_buffer.sample(
            int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
        )
        offline_batch = ds.sample(
            int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
        )

        batch = combine(offline_batch, online_batch)

        if "antmaze" in FLAGS.env_name:
            batch["rewards"] -= 1

        agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)
            
            wandb.log({f"evaluation/intervene_rate": np.mean(all_intervene[-FLAGS.eval_interval+1:])}, step=i + FLAGS.pretrain_steps)
            
            sampler_policy = RLPDSamplerPolicy(agent.actor)
            sparse_trajs = sparse_eval_sampler.sample(
                    sampler_policy,
                    FLAGS.eval_episodes, deterministic=False
                )
            avg_success = evaluate_policy(sparse_trajs,
                                            success_rate=True,
                                            success_function=lambda t: np.all(t['rewards'][-1:]>=10),
                                            )
            wandb.log({f"evaluation/avg_success": avg_success}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")
            
            if FLAGS.save_model:
                save_data = {'rlpd': agent}
                with open(os.path.join(model_dir, "model.pkl"), 'wb') as fout:
                    pickle.dump(save_data, fout)




if __name__ == "__main__":
    app.run(main)