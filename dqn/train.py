import logging
import wandb
wandb.init(project="dqn")

import dm_env
from bsuite.baselines import base

from .agent import DQN


def train(agent: base.Agent, env: dm_env.Environment, num_episodes: int) -> base.Agent:
    logging.info("Starting training agent {} on environment {}.\nThe scheduled number of episode is {}".format(agent, env, num_episodes))
    for episode in range(num_episodes):
        logging.info("Starting episode number {}/{}".format(episode, num_episodes - 1))
        wandb.log({"Episode": episode})
        # initialise environment
        timestep = env.reset()
        while not timestep.last():
            # policy
            action = agent.select_action(timestep)
            # step environment
            new_timestep = env.step(tuple(action))
            wandb.log({"Reward": new_timestep.reward})
            # update
            loss = agent.update(timestep, action, new_timestep)
            wandb.log({"Bellman MSE": loss})
            # prepare next
            timestep = new_timestep
    return agent
