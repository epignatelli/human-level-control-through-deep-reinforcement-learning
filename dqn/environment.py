from collections import deque
import gym
import dm_env
from bsuite.utils.gym_wrapper import DMEnvFromGym
import numpy as onp


class AtariEnv(DMEnvFromGym):
    def __init__(self, env_name, action_repeat):
        # public:
        self.action_repeat = action_repeat

        # private:
        self._base = super()
        self._base.__init__(gym.make(env_name))
        self._state_history = deque(maxlen=action_repeat)

    def step(self, action):
        # the agent selects an action only every k frames
        for _ in range(self.action_repeat):
            timestep = self._base.step(action)
            self._state_history.append(timestep.observation)
        return dm_env.TimeStep(
            timestep.step_type,
            timestep.reward,
            timestep.discount,
            onp.stack(self._state_history)
        )