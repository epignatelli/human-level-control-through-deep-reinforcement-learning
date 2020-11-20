from typing import NamedTuple
import jax
import jax.numpy as jnp


class Trajectory(NamedTuple):
    s0: jnp.ndarray
    a: int
    r: float
    s1: jnp.ndarray


def init_buffer(rng, env, size):
    buffer = []
    done = True
    for t in range(size):
        # reset if state is a terminal state
        if done:
            state = env.reset()
        # we generate the buffer according to a random policy
        action = int(jax.random.randint(rng, (1,), 0, env.action_space.n))
        # execute the action
        new_state, r, done, _ = env.step(action)
        # store trajectory (s, a, r, s')
        traj = Trajectory(s0=state, a=action, r=r, s1=new_state)
        buffer.append(traj)
        state = new_state
    return buffer


def sample_buffer(rng, buffer):
    idx = int(jax.random.randint(rng, (1,), 0, len(buffer)))
    return buffer[idx]
