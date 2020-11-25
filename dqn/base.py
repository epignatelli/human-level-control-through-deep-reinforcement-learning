from typing import Any, NamedTuple, Tuple
import functools
import jax.numpy as jnp

Params = Any
RNGKey = jnp.ndarray
Shape = Tuple[int, ...]


class Module(NamedTuple):
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]


def module(module_maker):
    @functools.wraps(module_maker)
    def fabricate_module(*args, **kwargs):
        init, apply = module_maker(*args, **kwargs)
        return Module(init, apply)

    return fabricate_module
