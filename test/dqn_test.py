import bsuite
import dm_env
from dqn import DQN


def test_dqn():
    n_actions = 10
    in_shape = (4, 84, 84)
    hparams = HParams()
    dqn = DQN(n_actions, in_shape, hparams)
    r = 0
    x = jax.random.normal(dqn.rng, (84, 84, 3))
    timestep = dm_env.TimeStep(dm_env.StepType.FIRST, r, dqn.hparams.discount, x)
    action = dqn.select_action(timestep)
    y = jax.random.normal(dqn.rng, (84, 84, 3))
    new_timestep = dm_env.TimeStep(dm_env.StepType.MID, 1.0, dqn.hparams.discount, y)
    dqn.update(timestep, action, new_timestep)
    return dqn
