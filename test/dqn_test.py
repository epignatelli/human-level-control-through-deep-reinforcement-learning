import bsuite
import dm_env
from dqn.agent import DQN


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
