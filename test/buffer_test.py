import logging
import dm_env
import numpy as onp
from dqn import ReplayBuffer


def test_buffer_replace():
    shape = (2, 2)
    capacity = 2
    buffer = ReplayBuffer(shape, capacity)
    for i in range(10):
        x = onp.ones(shape) * i
        a, r = i, i
        discount = 1.0
        timestep = dm_env.TimeStep(dm_env.StepType.FIRST, r, discount, x)
        buffer.add(timestep, a, timestep)
        logging.debug(
            "i: {},  r: {}, len(buffer): {}".format(i, buffer._r, len(buffer))
        )
    # make sure the buffer recycles if adding more elements than its capacity
    assert len(buffer) == 2
    # make sure the oldest elements are recycled
    assert onp.array_equal(
        buffer._s0,
        onp.array(
            [[[8.0, 8.0], [8.0, 8.0]], [[9.0, 9], [9.0, 9.0]]], dtype=onp.float32
        ),
    )
    assert onp.array_equal(buffer._r, onp.array([8.0, 9.0], dtype=onp.float32))
    assert onp.array_equal(buffer._a, onp.array([8.0, 9.0], dtype=onp.float32))
    # try sampling with n < len(buffer)
    batch = buffer.sample(1)
    assert len(batch[0]) == 1
    logging.debug(batch)
    # try sampling wiht n == len(buffer)
    batch = buffer.sample(2)
    assert len(batch[0]) == len(buffer)
    logging.debug(batch)
    # try sampling with n > len(buffer)
    batch = buffer.sample(3)
    assert len(batch[0]) == len(buffer)
    logging.debug(batch)
    return
