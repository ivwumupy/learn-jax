from collections.abc import Callable
from dataclasses import dataclass
import enum
import math
from typing import Optional

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
        self.b = nnx.Param(rngs.params.uniform((output_dim,)))

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
        use_scale: bool = False,
        use_bias: bool = False,
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
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: Optional[jax.Array] = None,
    use_scale: bool = True,
) -> jax.Array:
    """
    Compute the (scaled) dot-product attention.

        Attention(Q, K, V) = softmax(Q K^t / sqrt(d_k)) V

    Args:
        q: shape (...batch, query_count, qk_dim)
        k: shape (...batch, kv_count, qk_dim)
        v: shape (...batch, kv_count, v_dim)

        mask: shape (...batch, query_count, kv_count)

    Return:
        shape (...batch, query_count, v_dim)
    """
    assert q.shape[-1] == k.shape[-1]
    assert k.shape[-2] == v.shape[-2]
    qk = jnp.einsum("...ij,...kj->...ik", q, k)
    if use_scale:
        dk = k.shape[-1]
        qk *= lax.rsqrt(float(dk))
    if mask is not None:
        neg_inf = jnp.finfo(jnp.float32).min
        qk = jnp.where(mask, qk, neg_inf)
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

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        *,
        mask: Optional[jax.Array] = None,
    ):
        q1 = q @ self.W_q
        k1 = k @ self.W_k
        v1 = v @ self.W_v
        return generic_dot_product_attention(q1, k1, v1, mask, use_scale=self.use_scale)


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


def all_you_need_init(block_size: int, embed_dim: int):
    arr = []
    for pos in range(block_size):
        v = []
        for i in range(embed_dim // 2):
            t = 2 * i / embed_dim
            v.append(math.sin(pos / (block_size**t)))
            v.append(math.cos(pos / (block_size**t)))
        arr.append(v)
    return jnp.array(arr)


class StaticPositionEncoding(nnx.Module):
    def __init__(
        self,
        block_size: int,
        embed_dim: int,
        initializer: Callable[[int, int], jax.Array],
    ):
        self.A = initializer(block_size, embed_dim)

    def __call__(self, pos: jax.Array):
        return jnp.take(self.A, pos, axis=0)


class PositionEncodingStrategy(enum.Enum):
    NONE = enum.auto()
    LEARNED = enum.auto()
    ALL_YOU_NEED = enum.auto()
    ROPE = enum.auto()


@dataclass
class MicroLMConfig:
    vocab_size: int
    embed_dim: int
    qk_dim: int
    hidden_dim: int
    block_size: int
    layer_count: int

    position_encoding: PositionEncodingStrategy = PositionEncodingStrategy.ALL_YOU_NEED


class MicroLM(nnx.Module):
    attentions: list[DotProductAttention]
    feed_forwards: list[FeedForward]

    def __init__(
        self,
        config: MicroLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.token_embed = Embedding(config.vocab_size, config.embed_dim, rngs=rngs)

        match config.position_encoding:
            case PositionEncodingStrategy.NONE:
                self.pos_embed = None
            case PositionEncodingStrategy.LEARNED:
                self.pos_embed = Embedding(
                    config.block_size, config.embed_dim, rngs=rngs
                )
            case PositionEncodingStrategy.ALL_YOU_NEED:
                self.pos_embed = StaticPositionEncoding(
                    config.block_size, config.embed_dim, all_you_need_init
                )
            case _:
                raise Exception("unimplmented")

        self.embed_normalization = LayerNorm(config.embed_dim, rngs=rngs)

        self.attentions = []
        self.feed_forwards = []

        for _ in range(config.layer_count):
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

    def __call__(self, input: jax.Array, padding_mask: jax.Array):
        """
        Args:
            input: shape (...batch, block_size). an array of token ids
        """
        seq_len = input.shape[-1]
        assert seq_len == self.config.block_size
        assert seq_len == padding_mask.shape[-1]

        mask = padding_mask[:, jnp.newaxis] & padding_mask[..., jnp.newaxis]

        x = self.token_embed(input)

        if self.pos_embed is not None:
            x += self.pos_embed(jnp.arange(seq_len))

        for i in range(self.config.layer_count):
            x_norm = self.embed_normalization(x)
            x += self.attentions[i](x_norm, x_norm, x_norm, mask=mask)
            x_norm = self.embed_normalization(x)
            x += self.feed_forwards[i](x_norm)

        x_norm = self.embed_normalization(x)
        return self.lm_head(x_norm)
