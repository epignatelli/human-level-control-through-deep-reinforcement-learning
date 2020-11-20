import jax
import jax.numpy as jnp
from jax.experimental.stax import Dense, GeneralConv, Flatten, Relu, serial


Params = Any
RNGKey = jnp.ndarray
Shape = Tuple


class Module(NamedTuple):
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]


def module(module_maker):
    @functools.wraps(module_maker)
    def fabricate_module(*args, **kwargs):
        init, apply = module_maker(*args, **kwargs)
        return Module(init, apply)
    return fabricate_module


@module
def DQN(n_actions, input_format=("NCWH", "IWHO", "NCWH")):
    return serial(
        GeneralConv(input_format, 32, (8, 8), (4, 4), "VALID"),
        Relu,
        GeneralConv(input_format, 64, (4, 4), (2, 2), "VALID"),
        Relu,
        GeneralConv(input_format, 64, (3, 3), (1, 1), "VALID"),
        Relu,
        Flatten,
        Dense(512),
        Relu,
        Dense(n_actions)
    )


@functools.partial(jax.jit, static_argnums=0)
def forward(model, x, r, gamma, params_b, params_t):
    q_target = model.apply(params_t, x)
    q_behaviour = model.apply(params_b, x)
    y = r + gamma * jnp.max(q_target, axis=1)
    return jnp.mean(jnp.power((y.reshape(-1, 1) - q_behaviour), 2))


@functools.partial(jax.jit, static_argnums=0)
def backward(model, x, r, gamma, params_b, params_t):
    return jax.value_and_grad(forward, argnums=4)(model, x, r, gamma, params_b, params_t)


@functools.partial(jax.jit, static_argnums=(0, 1))
def update(model, optimiser, iteration, optimiser_state, x, r, gamma, params_b, params_t):
    loss, gradients = backward(model, x, r, gamma, params_b, params_t)
    return loss, optimiser.update_fn(iteration, gradients, optimiser_state)

@jax.jit
def preprocess(batch_x):
    # x is (n_frames, 210, 160, 3)
    target_shape = (batch_x.shape[0], batch_x.shape[1], 84, 84)
    luminance_mask = jnp.array([0.2126, 0.7152, 0.0722]).reshape(1, 1, 1, 1, 3)
    # get luminance
    y = jnp.sum(batch_x * luminance_mask, axis=-1)
    # resize
    s = jax.image.resize(y, target_shape, method="bilinear")
    return s
