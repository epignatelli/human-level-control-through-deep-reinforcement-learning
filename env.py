def init_environment(env_name, seed, max_episode_length):
    env = gym.make("{}-v4".format(env_name))
    env.seed = seed
    env.spec.max_episode_steps = max_episode_length
    return env
