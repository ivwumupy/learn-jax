import functools

import jax
import jax.numpy as jnp
from jax import lax


@functools.partial(jax.jit, static_argnames=["length", "temp"])
def generate_text(model, key, temp, length, initial):
    block_size = initial.shape[0]

    def scan_gen(carry, _):
        key, context, mask = carry
        logits = model(context, mask)
        n = jnp.count_nonzero(mask)
        key, subkey = jax.random.split(key)
        new_token = jax.random.categorical(subkey, logits[n - 1] / temp, shape=(1,))
        m = jnp.max(jnp.array([n, block_size - 1])) + 1
        new_mask = jnp.concatenate(
            [
                jnp.full((m,), True),
                jnp.full((block_size - m,), False),
            ]
        )
        context = jnp.concatenate([context[1:], new_token])
        return (key, context, new_mask), new_token

    _, new_tokens = lax.scan(scan_gen, (key, initial), (), length=length)
    return new_tokens


def generate_text2(model, key, temp, length, initial: jax.Array):
    """
    Args:
        initial: assumed to be padded
    """

    @jax.jit
    def run(input, mask):
        return model(input, mask)

    block_size = initial.shape[0]

    new_tokens = []

    context = initial

    for _ in range(length):
        # Compute the initial mask.
        non_pad = jnp.count_nonzero(context)

        mask = jnp.pad(
            jnp.full((non_pad,), True),
            (0, block_size - non_pad),
            mode="constant",
            constant_values=False,
        )

        logits = run(context, mask)
        key, subkey = jax.random.split(key)

        non_pad = jnp.count_nonzero(mask)

        new_token = jax.random.categorical(
            subkey, logits[non_pad - 1] / temp, shape=(1,)
        )
        new_tokens.append(new_token.tolist()[0])

        if non_pad == block_size:
            context = jnp.concatenate([context[1:], new_token])
        else:
            context = jnp.pad(
                jnp.concatenate([context[:non_pad], new_token]),
                (0, block_size - non_pad - 1),
                mode="constant",
                constant_values=0,
            )

    return new_tokens
