import dm_env
from bsuite.baselines import base

from .agent import DQN


def train(agent: base.Agent, env: dm_env.Environment, num_episodes: int) -> base.Agent:
    for episode in range(num_episodes):
        timestep = env.reset()
        while not timestep.last():
            # policy
            action = agent.select_action(timestep)
            # step environment
            new_timestep = env.step(action)
            # update
            agent.update(timestep, action, new_timestep)
            # prepare next
            timestep = new_timestep
    return agent
