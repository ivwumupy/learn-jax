from dataclasses import dataclass

from flax import nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import lax


class Linear(nnx.Module):
    """
    A linear transformation.

        y = xA + b
    """

    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        """
        Args:
            input_dim : dimension of the input vector
            output_dim : dimension of the output vector
        """
        self.A = nnx.Param(rngs.params.uniform((input_dim, output_dim)))
        self.b = nnx.Param(jnp.zeros(output_dim))

    def __call__(self, x: jax.Array):
        return x @ self.A + self.b


class Embedding(nnx.Module):
    """
    Map discrete tokens to an embedding space.

        token i --> A[i]
    """

    def __init__(self, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.A = nnx.Param(rngs.params.uniform((vocab_size, embed_dim)))

    def __call__(self, ids: jax.Array):
        return jnp.take(self.A.value, ids, axis=0)


class LayerNorm(nnx.Module):
    """
    Layer normalization. arXiv:1607.06450

        y = scale * (x - E[x]) / sqrt(Var[x] + epsilon) + bias
    """

    def __init__(
        self,
        input_dim: int,
        *,
        use_scale: bool = True,
        use_bias: bool = True,
        rngs: nnx.Rngs,
    ):
        if use_scale:
            self.scale = nnx.Param(rngs.params.uniform((input_dim,)))
        else:
            self.scale = None
        if use_bias:
            self.bias = nnx.Param(rngs.params.uniform((input_dim,)))
        else:
            self.bias = None
        self.epsilon = 1e-9

    def __call__(self, x: jax.Array):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        result = (x - mean) * lax.rsqrt(var + self.epsilon)
        if self.scale is not None:
            result *= self.scale.value
        if self.bias is not None:
            result += self.bias.value
        return result


def generic_dot_product_attention(
    q: jax.Array, k: jax.Array, v: jax.Array, *, use_scale: bool = True
) -> jax.Array:
    """
    Compute the (scaled) dot-product attention.

        Attention(Q, K, V) = softmax(Q K^t / sqrt(d_k)) V

    Args:
        q: shape (...batch, query_count, qk_dim)
        k: shape (...batch, kv_count, qk_dim)
        v: shape (...batch, kv_count, v_dim)

    Return:
        shape (...batch, query_count, v_dim)
    """
    assert q.shape[-1] == k.shape[-1]
    qk = jnp.einsum("...ij,...kj->...ik", q, k)
    if use_scale:
        dk = k.shape[-1]
        # TODO:
        qk *= lax.rsqrt(float(dk))
    s = jnn.softmax(qk, axis=-1)
    return jnp.einsum("...ij,...jk->...ik", s, v)


class DotProductAttention(nnx.Module):
    """
    (Scaled) dot-product attention. arXiv:1706.03762
    """

    def __init__(
        self,
        input_q_dim: int,
        input_k_dim: int,
        input_v_dim: int,
        output_dim: int,
        qk_dim: int,
        *,
        use_scale: bool = True,
        rngs: nnx.Rngs,
    ):
        self.use_scale = use_scale
        self.W_q = nnx.Param(rngs.params.uniform((input_q_dim, qk_dim)))
        self.W_k = nnx.Param(rngs.params.uniform((input_k_dim, qk_dim)))
        self.W_v = nnx.Param(rngs.params.uniform((input_v_dim, output_dim)))

    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array):
        q = q @ self.W_q
        k = k @ self.W_k
        v = v @ self.W_v
        return generic_dot_product_attention(q, k, v, use_scale=self.use_scale)


class FeedForward(nnx.Module):
    """
    A fully connected feed-forward network of depth 2, for using in transformers.

        y = ReLU(x W_1 + b_1) W_2 + b_2
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, *, rngs: nnx.Rngs
    ):
        self.W_1 = nnx.Param(rngs.params.uniform((input_dim, hidden_dim)))
        self.b_1 = nnx.Param(rngs.params.uniform((hidden_dim,)))
        self.W_2 = nnx.Param(rngs.params.uniform((hidden_dim, output_dim)))
        self.b_2 = nnx.Param(rngs.params.uniform((output_dim,)))

    def __call__(self, x: jax.Array):
        y = x @ self.W_1 + self.b_1
        y = jnn.relu(y)
        return y @ self.W_2 + self.b_2


@dataclass
class MicroLMConfig:
    vocab_size: int
    embed_dim: int
    qk_dim: int
    hidden_dim: int
    block_size: int
    layer_count: int


class MicroLM(nnx.Module):
    def __init__(
        self,
        config: MicroLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.token_embed = Embedding(config.vocab_size, config.embed_dim, rngs=rngs)
        self.pos_embed = Embedding(config.block_size, config.embed_dim, rngs=rngs)
        self.embed_normalization = LayerNorm(config.embed_dim, rngs=rngs)

        self.attentions = []
        self.feed_forwards = []

        for i in range(config.layer_count):
            self.attentions.append(
                DotProductAttention(
                    config.embed_dim,
                    config.embed_dim,
                    config.embed_dim,
                    config.embed_dim,
                    config.qk_dim,
                    rngs=rngs,
                )
            )
            self.feed_forwards.append(
                FeedForward(
                    config.embed_dim, config.embed_dim, config.hidden_dim, rngs=rngs
                )
            )
        self.lm_head = Linear(config.embed_dim, config.vocab_size, rngs=rngs)

    def __call__(self, x: jax.Array):
        seq_len = x.shape[-1]
        assert seq_len == self.config.block_size

        # Token id --> embedding
        x = self.token_embed(x)
        x += self.pos_embed(jnp.arange(seq_len))
        # normalize each embedded token

        for i in range(self.config.layer_count):
            x_norm = self.embed_normalization(x)
            x += self.attentions[i](x_norm, x_norm, x_norm)
            x_norm = self.embed_normalization(x)
            x += self.feed_forwards[i](x_norm)

        x_norm = self.embed_normalization(x)
        return self.lm_head(x_norm)
