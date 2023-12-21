from functools import partial
import cloudpickle as pickle
import os
import collections
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, Type

import numpy as np
import distrax
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from tensorflow_probability.substrates import jax as tfp
from flax import struct
from flax.training.train_state import TrainState
import flax.linen as nn

import d4rl
import gym
import gymnasium


from ..utils.utils import set_random_seed
from ..utils.jax_utils import extend_and_repeat, next_rng, JaxRNG


DataType = Union[np.ndarray, Dict[str, "DataType"]]
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
ModuleDef = Any
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = params.copy(add_or_replace={"Ensemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params

@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng

@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()

def update_target_network(main_params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda x, y: tau * x + (1.0 - tau) * y,
        main_params, target_params
    )

def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped

def load_model(ckpt_path):
    with open(ckpt_path, 'rb') as fin:
        checkpoint_data = pickle.load(fin)
    return checkpoint_data

def get_policy_from_model(env, checkpoint_data):
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = TanhGaussianPolicy(
                observation_dim, 
                action_dim, 
                arch='256-256',
                orthogonal_init=False,
                log_std_multiplier=1.0,
                log_std_offset=-1.0,
            )
    params = model.init(
            next_rng(model.rng_keys()),
            jnp.zeros((10, observation_dim))
        )
    params = checkpoint_data['sac']._train_states['policy'].params

    sampler_policy = SamplerPolicy(model, params)
    return sampler_policy

def evaluate_policy(trajs, success_rate=True, success_function=None, log=False):
    if success_rate:
        successes = list(map(success_function, trajs))
        eval_n_trajs = len(trajs)
        mean_success_rates = np.mean(successes)
        if log:
            print(f"Success rate over {eval_n_trajs} trajectories: ", mean_success_rates)
        return mean_success_rates
    else:
        average_returns_sum = list(map(lambda t: sum(t['rewards']), trajs))
        mean_returns = np.mean(average_returns_sum)
        if log:
            print(f"Average returns over {eval_n_trajs} trajectories: ", mean_returns)
        return mean_returns


class SamplerPolicy(object):
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=('self', 'deterministic'))
    def act(self, params, rng, observations, deterministic):
        return self.policy.apply(
            params, observations, deterministic, repeat=None,
            rngs=JaxRNG(rng)(self.policy.rng_keys())
        )

    def __call__(self, observations, deterministic=False):
        actions, _ = self.act(self.params, next_rng(), observations, deterministic=deterministic)

        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class RandomSamplerPolicy(object):
    def __init__(self, env):
        self.env = env

    def __call__(self, observations, deterministic=False):
        actions = self.env.action_space.sample()
        return actions


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param('value', lambda x: self.init_value)

    def __call__(self):
        return self.value
    

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


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


class MLP_RLPD(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:

        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=nn.initializers.xavier_uniform(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=nn.initializers.xavier_uniform())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x


class MLPResNetV2Block(nn.Module):
    """MLPResNet block."""

    features: int
    act: Callable

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.LayerNorm()(x)
        y = self.act(y)
        y = nn.Dense(self.features)(y)
        y = nn.LayerNorm()(y)
        y = self.act(y)
        y = nn.Dense(self.features)(y)

        if residual.shape != y.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + y


class MLPResNetV2(nn.Module):
    """MLPResNetV2."""

    num_blocks: int
    features: int = 256
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.features)(x)
        for _ in range(self.num_blocks):
            x = MLPResNetV2Block(self.features, act=self.act)(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split('-')]
        for h in hidden_sizes:
            if self.orthogonal_init:
                x = nn.Dense(
                    h,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros
                )(x)
            else:
                x = nn.Dense(h)(x)
            
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.orthogonal(1e-2),
                bias_init=jax.nn.initializers.zeros
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(
                    1e-2, 'fan_in', 'uniform'
                ),
                bias_init=jax.nn.initializers.zeros
            )(x)
        return output

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise')


class FullyConnectedQFunction(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False
    layer_norm: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = FullyConnectedNetwork(output_dim=1, arch=self.arch, orthogonal_init=self.orthogonal_init, layer_norm=self.layer_norm)(x)
        return jnp.squeeze(x, -1)

    @nn.nowrap
    def rng_keys(self):
        return ('params', )


class StateActionValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(outputs)

        return jnp.squeeze(value, -1)


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        return critic1, critic2


class TanhGaussianPolicy(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = '256-256'
    orthogonal_init: bool = False
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0
    use_tanh: bool = True

    def setup(self):
        self.base_network = FullyConnectedNetwork(
            output_dim=2 * self.action_dim, arch=self.arch, orthogonal_init=self.orthogonal_init
        )
        self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
        self.log_std_offset_module = Scalar(self.log_std_offset)

    def log_prob(self, observations, actions, rewards=None):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        if self.use_tanh:
            action_distribution = distrax.Transformed(
                action_distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        def get_weight(rewards):
            weight = jnp.exp(rewards) / (1  + jnp.exp(-1))
            return weight 
        if not rewards is None:
            weight = get_weight(rewards)
            return action_distribution.log_prob(actions) * weight
        else:
            return action_distribution.log_prob(actions)

    def __call__(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        if self.use_tanh:
            action_distribution = distrax.Transformed(
                action_distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        if deterministic:
            samples = mean
            if self.use_tanh:
                samples = jnp.tanh(samples)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(seed=self.make_rng('noise'))

        return samples, log_prob

    @nn.nowrap
    def rng_keys(self):
        return ('params', 'noise')


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_stds_val: float = 0
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.normal(self.log_stds_val),
                                  (self.action_dim, )) # TODO: fix 0.05, 0.01, 0.15

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist
