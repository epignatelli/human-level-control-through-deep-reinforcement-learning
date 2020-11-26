import logging
from dqn import AtariEnv, ATARI_ENV_LIST


def test_atarienv():
    for env_name in ATARI_ENV_LIST:
        env = AtariEnv(env_name, 4)
        env.reset()
        timestep = env.step(0)
        assert len(timestep.observation) == 4
        logging.debug(env_name + " successfully initialised")
        env.close()



test_atarienv()