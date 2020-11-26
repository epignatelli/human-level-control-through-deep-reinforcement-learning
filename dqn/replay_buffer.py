from typing import NamedTuple, Union
from collections import deque
import functools
import logging
import numpy as onp
import jax


class Transition(NamedTuple):
    s: onp.ndarray
    a: onp.ndarray
    r: onp.ndarray
    ns: onp.ndarray


class ReplayBuffer:
    def __init__(self, observation_shape, capacity, seed=0):
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
                timestep.observation,
                action,
                timestep.reward,
                new_timestep.observations
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
        s, a, r, ns = [], [], [], []
        batch = [[], [], [], []]
        for idx in indices:
            _s, _a, _r, _ns = self._data[idx]
            traj = self._data[idx]
            for i, item in enumerate(traj):
                batch[i].append(item)
        s = onp.stack(s)
        a = onp.stack(a)
        r = onp.stack(r)
        ns = onp.stack(ns)
        return (s, a, r, ns)
