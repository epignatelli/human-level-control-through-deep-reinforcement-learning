from collections import deque
import gym
import dm_env
from bsuite.utils.gym_wrapper import DMEnvFromGym
import numpy as onp


class AtariEnv(DMEnvFromGym):
    def __init__(self, env_name: str, action_repeat: int):
        # public:
        self.action_repeat = action_repeat

        # private:
        self._base = super()
        self._base.__init__(gym.make(env_name))
        self._state_history = deque(maxlen=action_repeat)

    def step(self, action: int) -> dm_env.TimeStep:
        # the agent selects an action only every k frames
        for _ in range(self.action_repeat):
            timestep = self._base.step(action)  # type: dm_env.TimeStep
            self._state_history.append(timestep.observation)
        return timestep._replace(observation=onp.stack(self._state_history))


ATARI_ENV_LIST = list(
    map(
        lambda x: x + "-v4",
        [
            "Alien",
            "Amidar",
            "Assault",
            "Asterix",
            "Asteroids",
            "Atlantis",
            "BankHeist",
            "BattleZone",
            "BeamRider",
            "Bowling",
            "Boxing",
            "Breakout",
            "Centipede",
            "ChopperCommand",
            "CrazyClimber",
            "DemonAttack",
            "DoubleDunk",
            "Enduro",
            "FishingDerby",
            "Freeway",
            "Frostbite",
            "Gopher",
            "Gravitar",
            "Hero",
            "IceHockey",
            "Jamesbond",
            "Kangaroo",
            "Krull",
            "KungFuMaster",
            "MontezumaRevenge",
            "MsPacman",
            "NameThisGame",
            "Pong",
            "PrivateEye",
            "Qbert",
            "Riverraid",
            "RoadRunner",
            "Robotank",
            "Seaquest",
            "SpaceInvaders",
            "StarGunner",
            "Tennis",
            "TimePilot",
            "Tutankham",
            "UpNDown",
            "Venture",
            "VideoPinball",
            "WizardOfWor",
            "Zaxxon",
        ],
    )
)
