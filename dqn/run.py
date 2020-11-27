import logging
import wandb

import dm_env
from bsuite.baselines import base

from .agent import DQN


def run(
    agent: base.Agent,
    env: dm_env.Environment,
    num_episodes: int,
    eval_mode: bool = False,
) -> base.Agent:
    wandb.init(project="dqn")
    logging.info(
        "Starting {} agent {} on environment {}.\nThe scheduled number of episode is {}".format(
            "evaluating" if eval_mode else "training", agent, env, num_episodes
        )
    )
    for episode in range(num_episodes):
        print(
            "Starting episode number {}/{}\t\t\t".format(episode, num_episodes - 1),
            end="\r",
        )
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
            if not eval_mode:
                loss = agent.update(timestep, action, new_timestep)
                wandb.log({"Bellman MSE": float(loss)})
                wandb.log({"Iteration": agent.iteration})
            # prepare next
            timestep = new_timestep
    return agent
