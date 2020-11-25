import logging
import numpy as onp
import jax


class ReplayBuffer:
    def __init__(self, observation_shape, capacity, seed=0):
        # public:
        self.capacity = capacity
        self.observation_shape = observation_shape
        onp.random.seed(seed)

        # private:
        self._s0 = onp.empty((capacity, *observation_shape), dtype=onp.float32)
        self._a = onp.empty((capacity,), dtype=onp.int32)
        self._r = onp.empty((capacity,), dtype=onp.float32)
        self._s1 = onp.empty((capacity, *observation_shape), dtype=onp.float32)
        self._current_idx = 0
        self._full = False

    def __len__(self):
        return self.capacity if self._full else self._current_idx

    def __getitem__(self, idx):
        s0 = self._s0[idx]
        a = self._a[idx]
        r = self._r[idx]
        s1 = self._s1[idx]
        return (s0, a, r, s1)

    def add(self, timestep, action, new_timestep):
        self._s0[self._current_idx] = timestep.observation
        self._a[self._current_idx] = action
        self._r[self._current_idx] = timestep.reward
        self._s1[self._current_idx] = new_timestep.observation
        self._current_idx += 1
        # if buffer is full, start replacing older transitions
        if self._current_idx >= self.capacity:
            self._current_idx = 0
            self._full = True
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
        s0 = onp.stack([self._s0[i] for i in indices])
        a = onp.stack([self._a[i] for i in indices])
        r = onp.stack([self._r[i] for i in indices])
        s1 = onp.stack([self._s1[i] for i in indices])
        return (s0, a, r, s1)
