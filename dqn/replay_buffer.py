from typing import NamedTuple, Union
from collections import deque
import logging
import numpy as onp


class Transition(NamedTuple):
    s: onp.ndarray
    a: onp.ndarray
    r: onp.ndarray
    ns: onp.ndarray


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        # public:
        self.capacity = capacity

        # private:
        self._data = deque(maxlen=capacity)
        onp.random.seed(seed)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def add(self, timestep, action, new_timestep):
        self._data.append(
            Transition(
                timestep.observation, action, timestep.reward, new_timestep.observation
            )
        )
        return

    def sample(self, n):
        high = len(self) - n
        if high <= 0:
            logging.warning(
                "The buffer contains less elements than requested: {} <= {}\n"
                "Returning all the available elements".format(len(self), n)
            )
            indices = range(len(self))
        else:
            indices = onp.random.randint(0, high, size=n)

        return tuple(zip(*(map(lambda idx: self._data[idx], indices))))
