"""CODE ADAPTED FROM ILYA KOSTRIKOV'S IQL (https://github.com/ikostrikov/implicit_q_learning)""" 

from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import collections
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn

from tqdm import tqdm

from ..models.model import (
    ValueCritic, DoubleCritic, NormalTanhPolicy, Model
)
from ..utils.jax_utils import extend_and_repeat, next_rng, JaxRNG


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


def get_iql_policy_from_model(env, checkpoint_data):
    sampler_policy = IQLSamplerPolicy(checkpoint_data['iql'].actor)
    return sampler_policy


class IQLSamplerPolicy(object):

    def __init__(self, actor):
        self.actor=actor
        rng = jax.random.PRNGKey(24)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        self.rng = rng

    def __call__(self, observations, deterministic=False):
        actions = self.sample_actions(observations)

        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = sample_actions_actor(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)


@partial(jax.jit, static_argnames=('actor_def', 'distribution'))
def _sample_actions(rng: PRNGKey,
                    actor_def: nn.Module,
                    actor_params: Params,
                    observations: np.ndarray,
                    temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions_actor(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations,
                           temperature)
       

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, batch, temperature)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(critic: Model, value: Model, batch: Batch,
             expectile: float) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1, q2 = critic(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def awr_update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * temperature)
    # exp_a = jnp.exp(temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})

        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()
        

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v, 'exp_a': exp_a}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


class IQLLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 policy_weight_decay: float = 0.0,
                 log_stds=0.0,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine"):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.log_stds = log_stds

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = NormalTanhPolicy(hidden_dims,
                                    action_dim,
                                    log_std_scale=1e-3,
                                    log_std_min=-5.0,
                                    dropout_rate=dropout_rate,
                                    state_dependent_std=False,
                                    log_stds_val=self.log_stds,
                                    tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            if policy_weight_decay != 0:
                optimiser = optax.adamw(learning_rate=actor_lr, weight_decay=policy_weight_decay)
            else:
                optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = sample_actions_actor(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.temperature)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info