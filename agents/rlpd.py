"""CODE ADAPTED FROM ILYA KOSTRIKOV'S RLPD (https://github.com/ikostrikov/rlpd)""" 

import copy
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import flax
import gym
import jax
import numpy as np
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from flax.core.frozen_dict import unfreeze, freeze
import flax.linen as nn

from ..utils.dataset_utils import DatasetDict
from ..models.distributions import TanhNormal
from ..models.model import (
    MLP_RLPD,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_ensemble,
)

from flax import struct
from flax.training.train_state import TrainState

DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]


def get_rlpd_policy_from_model(env, checkpoint_data):
    sampler_policy = RLPDSamplerPolicy(checkpoint_data['rlpd'].actor)
    return sampler_policy


class RLPDSamplerPolicy(object):

    def __init__(self, actor):
        self.actor=actor
        
        rng = jax.random.PRNGKey(24)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        self.rng = rng
        self.agent = Agent(self.actor, self.rng)

    def __call__(self, observations, deterministic=False):
        actions = self.sample_actions(observations)

        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)

    def sample_actions(self,
                       observations: np.ndarray) -> jnp.ndarray:
        actions, rng = _sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    

@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return np.asarray(actions), self.replace(rng=new_rng)

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)

# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


class SACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        preload_actor_params: Optional[dict] = None,
        preload_critic_params: Optional[dict] = None,
        preload_target_critic_params: Optional[dict] = None,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(
            MLP_RLPD, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        if not preload_actor_params is None:
            actor_params = preload_actor_params
            actor_params = unfreeze(actor_params)
            actor_params['MLP_RLPD_0'] = actor_params['MLP_0']
            del actor_params['MLP_0']
            actor_params = freeze(actor_params)
        else:
            actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP_RLPD,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)

        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        
        if not preload_critic_params is None:
            critic_params = preload_critic_params

            critic_params = unfreeze(critic_params)
            critic_params['VmapStateActionValue_0']['MLP_RLPD_0'] = critic_params['VmapStateActionValue_0']['MLP_0']
            del critic_params['VmapStateActionValue_0']['MLP_0']
            critic_params = freeze(critic_params)

            target_critic_params = preload_target_critic_params

            target_critic_params = unfreeze(target_critic_params)
            target_critic_params['VmapStateActionValue_0']['MLP_RLPD_0'] = target_critic_params['VmapStateActionValue_0']['MLP_0']
            del target_critic_params['VmapStateActionValue_0']['MLP_0']
            target_critic_params = freeze(target_critic_params)
        else:
            critic_params = critic_def.init(critic_key, observations, actions)["params"]
            target_critic_params = copy.deepcopy(critic_params)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=target_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}
