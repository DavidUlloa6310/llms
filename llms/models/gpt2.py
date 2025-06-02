from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072
    layer_norm_epsilon: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    initializer_range: float = 0.01
    dtype: jnp.dtype = jnp.bfloat16  # dtype used in computations
    param_dtype: jnp.dtype = jnp.float32  # dtype used to store parameters


class MHSelfAttention(nnx.Module):

    def __init__(self, rngs):
        self.config = GPT2Config()
        config = self.config

        self.qkv = nnx.Linear(
            config.n_embd,
            config.n_embd * 3,
            rngs=rngs,
            use_bias=True,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            param_dtype=config.param_dtype
        )

        self.out = nnx.Linear(
            config.n_embd,
            config.n_embd,
            rngs=rngs,
            use_bias=True,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            bias_init=nnx.initializers.zeros,
            param_dtype=config.param_dtype,
            dtype=config.dtype,
        )

        self.dropout = nnx.Dropout(rate=config.attn_dropout_rate, rngs=rngs)

    def __call__(self, hidden_states):
        """
        hidden_states: [batch_size, sequence_length, n_embd]
        """
        config = self.config
        batch_size, seq_length, hidden_dim = hidden_states.shape
        assert hidden_dim == config.n_embd

        qkv = self.qkv(hidden_states).reshape(batch_size, seq_length, 3,
                                              config.n_head, config.n_embd // config.n_head)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(axis=2)  # [batch_size, seq_length, n_head, head_dim]
        k = k.squeeze(axis=2)
        v = v.squeeze(axis=2)

        # Transpose for (batch, head, seq, head_dim)
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        causal_mask = jnp.tril(
            jnp.ones((seq_length, seq_length), dtype=config.dtype))
        causal_mask = causal_mask.reshape(1, 1, seq_length, seq_length)

        # Scaled dot-product attention
        inv_sqrt_d = jnp.asarray(
            1.0 / jnp.sqrt(k.shape[-1]), dtype=config.dtype)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * inv_sqrt_d

        # Apply the mask
        neg_inf = jnp.full_like(attn_weights, -1e4, dtype=config.dtype)
        attn_weights = jnp.where(causal_mask == 0, neg_inf, attn_weights)

        attn_probs = nnx.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_probs)

        # Multiply by value matrix
        attn_output = jnp.matmul(attn_probs, v)

        # Revert transpose
        attn_output = attn_output.transpose((0, 2, 1, 3))

        # Combine heads
        attn_output = attn_output.reshape(
            batch_size, seq_length, config.n_embd)

        # Final linear projection
        return self.out(attn_output)


class MLP(nnx.Module):

    def __init__(self, rngs):
        self.config = GPT2Config()
        config = self.config

        self.dense_1 = nnx.Linear(
            config.n_embd,
            config.n_inner,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )

        self.dense_2 = nnx.Linear(
            config.n_inner,
            config.n_embd,
            rngs=rngs,
            kernel_init=nnx.initializers.normal(config.initializer_range),
            bias_init=nnx.initializers.zeros,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )

        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, hidden_states):
        config = self.config

        hidden_states = hidden_states.astype(config.dtype)

        hidden = self.dense_1(hidden_states)
        hidden = nnx.gelu(hidden)

        hidden = self.dense_2(hidden)

        return self.dropout(hidden)


class TransformerBlock(nnx.Module):
    def __init__(self, rngs):
        self.config = GPT2Config()
        config = self.config

        self.layer_norm_1 = nnx.LayerNorm(
            num_features=config.n_embd,
            rngs=rngs,
            epsilon=config.layer_norm_epsilon,
            dtype=config.dtype,
            param_dtype=config.param_dtype
        )

        self.attention = MHSelfAttention(rngs=rngs)

        self.layer_norm_2 = nnx.LayerNorm(
            num_features=config.n_embd,
            rngs=rngs,
            epsilon=config.layer_norm_epsilon,
            dtype=config.dtype,
            param_dtype=config.param_dtype
        )

        self.mlp = MLP(rngs=rngs)

        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, hidden_states):
        config = self.config

        hidden_states = hidden_states.astype(config.dtype)

        attn_ln = self.layer_norm_1(hidden_states)
        attn_out = self.attention(attn_ln)
        attn_out = self.dropout(attn_out)

        x = hidden_states + attn_out

        mlp_ln = self.layer_norm_2(x)
        mlp_out = self.mlp(mlp_ln)

        # Residual
        x = x + mlp_out

        return x


class GPT2(nnx.Module):

    def __init__(self, rngs):
        self.config = GPT2Config()
        config = self.config

        self.wte = nnx.Embed(
            rngs=rngs,
            num_embeddings=config.vocab_size,
            features=config.n_embd,
            embedding_init=nnx.initializers.normal(
                config.initializer_range),
            param_dtype=config.param_dtype,
            dtype=config.dtype
        )

        self.wpe = nnx.Param(
            nnx.initializers.normal(stddev=config.initializer_range)(
                rngs.params(),
                (config.max_position_embeddings, config.n_embd),
                config.param_dtype
            )
        )

        self.blocks = [TransformerBlock(rngs=rngs)
                       for _ in range(self.config.n_layer)]

        self.layer_norm = nnx.LayerNorm(
            num_features=config.n_embd,
            epsilon=config.layer_norm_epsilon,
            param_dtype=config.param_dtype,
            dtype=config.dtype,
            rngs=rngs
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, input_ids):
        # Input embedding
        position_ids = jnp.arange(input_ids.shape[1])[None, :]
        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe[position_ids, :].astype(self.config.dtype)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(
            hidden_states)

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # Output logits by tying embeddings
        # We transpose the embedding matrix
        wte_tied = self.wte.embedding.T.astype(
            self.config.dtype
        )  # from [vocab_size, n_embd] to [n_embd, vocab_size]
        logits = jnp.einsum("bld,dk->blk", hidden_states, wte_tied)
        return logits
