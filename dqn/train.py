import gym
import dm_env
from .dqn import DQN


def env_from_name(env_name, version=4, seed=0):
    env = gym.make("{}-v{}".format(env_name.capitalize(), version))
    env.seed = seed
    return env


def train(env_name, in_shape, hparams, num_episodes):
    env = env_from_name(env_name)
    agent = DQN(env.action_space.n, in_shape, hparams)

    for episode in range(num_episodes):
        observation = env.reset()
        timestep = dm_env.TimeStep(dm_env.StepType.FIRST, None, hparams.discount, observation)
        while not timestep.last():
            # policy
            action = agent.select_action(timestep)

            # step environment
            new_observation, reward, done, _ = env.step(action)
            step_type = dm_env.StepType.LAST if done else dm_env.StepType.MID
            new_timestep = dm_env.TimeStep(step_type, reward, hparams.discount, new_observation)
            # update
            agent.update(timestep, action, new_timestep)
            # prepare next
            timestep = new_timestep
    return agent