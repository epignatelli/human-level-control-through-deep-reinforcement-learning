from typing import Tuple, NamedTuple, Callable, Any
import functools

import jax
import jax.numpy as jnp
from jax.experimental.stax import Conv, Dense, Relu, serial, GeneralConv, Flatten
from jax.experimental.optimizers import rmsprop_momentum

import dm_env
from bsuite.baselines import base

from .base import module, Optimiser
from .hparams import HParams
from .replay_buffer import ReplayBuffer


Shape = Tuple[int, ...]


class DQN(base.Agent):
    def __init__(
        self, n_actions: int, in_shape: Shape, hparams: HParams, seed: int = 0
    ):
        # differentiable:
        @module
        def network(n_actions, input_format=("NCWH", "IWHO", "NCWH")):
            return serial(
                GeneralConv(input_format, 32, (8, 8), (4, 4), "VALID"),
                Relu,
                GeneralConv(input_format, 64, (4, 4), (2, 2), "VALID"),
                Relu,
                GeneralConv(input_format, 64, (3, 3), (1, 1), "VALID"),
                Relu,
                Flatten,
                Dense(512),
                Relu,
                Dense(n_actions),
            )

        self.network = network

        def forward(model, x, r, discount, params_online, params_target):
            q_target = model.apply(params_target, x)
            q_behaviour = model.apply(params_online, x)
            # get the q target
            y = (r + discount * jnp.max(q_target, axis=1)).reshape(-1, 1)
            # clip the prediction error in the interval [-1, 1]
            error = jnp.clip(y - q_behaviour, -1, 1)
            return jnp.mean(jnp.power((error), 2))

        self.forward = jax.jit(forward, static_argnums=0)

        def backward(model, x, r, discount, params_online, params_target):
            return jax.value_and_grad(forward, argnums=4)(
                model, x, r, discount, params_online, params_target
            )

        self.backward = jax.jit(backward, static_argnums=0)

        def sgd_step(
            model,
            optimiser,
            iteration,
            optimiser_state,
            x,
            r,
            discount,
            params_online,
            params_target,
        ):
            loss, gradients = backward(
                model, x, r, discount, params_online, params_target
            )
            return loss, optimiser.update_fn(iteration, gradients, optimiser_state)

        self.sgd_step = jax.jit(sgd_step, static_argnums=(0, 1))

        def preprocess(x):
            # get luminance
            luminance_mask = jnp.array([0.2126, 0.7152, 0.0722]).reshape(1, 1, 1, 3)
            y = jnp.sum(x * luminance_mask, axis=-1).squeeze()
            # resize, batch_x is (batch_size, n_frames, 210, 160)
            target_shape = (*x.shape[:-3], 84, 84)
            s = jax.image.resize(y, target_shape, method="bilinear")
            return s

        self.preprocess = jax.jit(preprocess)
        self.preprocess_batch = jax.jit(jax.vmap(preprocess))

        # public:
        self.n_actions = n_actions
        self.hparams = hparams
        self.epsilon = hparams.initial_exploration
        self.replay_buffer = ReplayBuffer(in_shape, hparams.replay_memory_size)
        self.online_network = network(n_actions)
        self.target_network = network(n_actions)
        self.rng = jax.random.PRNGKey(seed)

        # private:
        self._iteration = 0
        self._online_params = self.online_network.init(self.rng, (-1, *in_shape))[1]
        self._target_params = self.target_network.init(self.rng, (-1, *in_shape))[1]
        self._optimiser = Optimiser(
            *rmsprop_momentum(
                step_size=hparams.learning_rate,
                gamma=hparams.squared_gradient_momentum,
                momentum=hparams.gradient_momentum,
                eps=hparams.min_squared_gradient,
            )
        )
        self._optimiser_state = self._optimiser.init(self._online_params)
        return

    def anneal_epsilon(self):
        x0, y0 = (self.hparams.replay_start, self.hparams.initial_exploration)
        x1, y1 = (
            self.hparams.replay_start + self.hparams.final_exploration_frame,
            self.hparams.final_exploration,
        )
        x = self._iteration
        y = ((y1 - y0) * (x - x0) / (x1 - x0)) + y0
        return y

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Policy function: maps the current observation/state to an action
        following an epsilon-greedy policy
        """
        # return random action with epsilon probability
        if jax.random.uniform(self.rng, (1,)) < self.epsilon:
            return jax.random.randint(self.rng, (1,), 0, self.n_actions)

        state = timestep.observation[None, ...]  # batching
        q_values = self.forward(self._online_params, state)
        # e-greedy
        action = jnp.argmax(q_values, axis=-1)
        return action

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ) -> None:
        # preprocess observations
        timestep = timestep._replace(observation=self.preprocess(timestep.observation))
        new_timestep = new_timestep._replace(
            observation=self.preprocess(new_timestep.observation)
        )

        # reward clipping
        reward = (
            -1.0
            if timestep.reward < -1.0
            else 1.0
            if timestep.reward > 1.0
            else timestep.reward
        )
        timestep = timestep._replace(reward=reward)

        # add experience to replay buffer
        self.replay_buffer.add(timestep, action, new_timestep)
        # if replay buffer is smaller than the minimum size, there is nothing else to do
        if len(self.replay_buffer) < self.hparams.replay_start:
            return

        # the exploration parameter is linearly interpolated to the end value
        self.epsilon = self.anneal_epsilon()

        # update the online parameters only every n interations
        if self._iteration % self.hparams.update_frequency:
            return
        transition_batch = self.replay_buffer.sample(self.hparams.batch_size)
        loss, self._optimiser_state = self.sgd_step(
            self.online_network,
            self._optimiser,
            self._iteration,
            self._optimiser_state,
            transition_batch[0],  # observation
            transition_batch[2],  # reward
            self.hparams.discount,
            self._online_params,
            self._target_params,
        )
        self._online_params = self._optimiser.params(self._optimiser_state)

        # update the target network parameters every n step
        if self._iteration % self.hparams.target_network_update_frequency == 0:
            self._target_params = self._online_params
        return
