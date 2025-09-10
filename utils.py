import functools

import jax
import jax.numpy as jnp
from jax import lax


@functools.partial(jax.jit, static_argnames=["length", "temp"])
def generate_text(model, key, temp, length, initial):
    def scan_gen(carry, _):
        key, context = carry
        logits = model(context)
        key, subkey = jax.random.split(key)
        new_token = jax.random.categorical(subkey, logits[-1] / temp, shape=(1,))
        context = jnp.concatenate([context[1:], new_token])
        return (key, context), new_token

    _, new_tokens = lax.scan(scan_gen, (key, initial), (), length=length)
    return new_tokens
