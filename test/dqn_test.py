import gym
import dm_env
from bsuite.utils.gym_wrapper import DMEnvFromGym

import jax
from dqn import DQN, HParams


def test_dqn():
    n_actions = 10
    in_shape = (4, 84, 84)
    hparams = HParams()
    agent = DQN(n_actions, in_shape, hparams)
    r = 0
    x = jax.random.normal(agent.rng, (84, 84, 3))
    timestep = dm_env.TimeStep(dm_env.StepType.FIRST, r, agent.hparams.discount, x)
    action = agent.select_action(timestep)
    y = jax.random.normal(agent.rng, (84, 84, 3))
    new_timestep = dm_env.TimeStep(dm_env.StepType.MID, 1.0, agent.hparams.discount, y)
    agent.update(timestep, action, new_timestep)
    return agent


def test_epsilon_annealing():
    gym_env = gym.make("Pong-v4")
    env = DMEnvFromGym(gym_env)
    in_shape = (4, 84, 84)
    hparams = HParams(
        replay_start=10,
        final_exploration_frame=20,
        initial_exploration=10.0,
        final_exploration=20.0,
    )
    agent = DQN(env.action_spec().num_values, in_shape, hparams, 0)

    agent.iteration = 10
    assert round(agent.anneal_epsilon(), 1) == 10.0

    agent.iteration = 20
    assert round(agent.anneal_epsilon(), 1) == 20.0

    agent.iteration = 15
    assert round(agent.anneal_epsilon(), 1) == 15.0
