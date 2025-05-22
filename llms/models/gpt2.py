from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


def assert_finite(tag: str, x: jax.Array):
    bad = ~jnp.isfinite(x)

    def _raise(tensor):
        jax.debug.print("{tag}: non-finite detected", tag=tag)
        # you can break or throw after the print; returning `tensor`
        # keeps the pytree structure identical
        return tensor

    # both branches now return `x` (the same array shape/dtype)
    _ = jax.lax.cond(jnp.any(bad), _raise, lambda tensor: tensor, operand=x)


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
    compute_dtype: jnp.dtype = jnp.float32  # dtype used to store parameters


class MHSelfAttention(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        """
        hidden_states: [batch_size, sequence_length, n_embd]
        """
        cfg = self.config
        batch_size, seq_length, hidden_dim = hidden_states.shape
        assert hidden_dim == cfg.n_embd

        qkv = nn.Dense(
            cfg.n_embd * 3,
            use_bias=True,
            kernel_init=nn.initializers.normal(cfg.initializer_range),
            bias_init=nn.initializers.zeros,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype
        )(hidden_states)

        qkv = qkv.reshape(batch_size, seq_length, 3,
                          cfg.n_head, cfg.n_embd // cfg.n_head)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(axis=2)  # [batch_size, seq_length, n_head, head_dim]
        k = k.squeeze(axis=2)
        v = v.squeeze(axis=2)

        # Transpose for (batch, head, seq, head_dim)
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        causal_mask = jnp.tril(
            jnp.ones((seq_length, seq_length), dtype=cfg.dtype))
        causal_mask = causal_mask.reshape(1, 1, seq_length, seq_length)

        # Scaled dot-product attention
        inv_sqrt_d = jnp.asarray(
            1.0 / jnp.sqrt(k.shape[-1]), dtype=cfg.dtype)

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * inv_sqrt_d

        # Apply the mask
        neg_inf = jnp.full_like(attn_weights, -1e4, dtype=cfg.dtype)
        attn_weights = jnp.where(causal_mask == 0, neg_inf, attn_weights)

        attn_probs = nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(rate=cfg.attn_dropout_rate)(
            attn_probs, deterministic=deterministic)

        # Multiply by value matrix
        attn_output = jnp.matmul(attn_probs, v)

        # Revert transpose
        attn_output = attn_output.transpose((0, 2, 1, 3))

        # Combine heads
        attn_output = attn_output.reshape(batch_size, seq_length, cfg.n_embd)

        # Final linear projection
        out = nn.Dense(
            cfg.n_embd,
            use_bias=True,
            kernel_init=nn.initializers.normal(cfg.initializer_range),
            bias_init=nn.initializers.zeros,
            param_dtype=cfg.param_dtype,
            dtype=cfg.dtype,
        )(attn_output)

        return out


class MLP(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        cfg = self.config

        hidden_states = hidden_states.astype(cfg.dtype)

        hidden = nn.Dense(
            cfg.n_inner,
            kernel_init=nn.initializers.normal(cfg.initializer_range),
            bias_init=nn.initializers.zeros,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype
        )(hidden_states)
        hidden = jnp.clip(hidden, -10, 10)
        hidden = nn.gelu(hidden)

        hidden = nn.Dense(
            cfg.n_embd,
            kernel_init=nn.initializers.normal(cfg.initializer_range),
            bias_init=nn.initializers.zeros,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype
        )(hidden)

        hidden = nn.Dropout(rate=cfg.dropout_rate)(
            hidden, deterministic=deterministic)
        return hidden


class TransformerBlock(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, hidden_states, deterministic: bool = True):
        cfg = self.config

        hidden_states = hidden_states.astype(cfg.dtype)

        attn_ln = nn.LayerNorm(
            epsilon=cfg.layer_norm_epsilon,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype
        )(hidden_states)
        attn_out = MHSelfAttention(cfg)(attn_ln, deterministic=deterministic)
        attn_out = nn.Dropout(rate=cfg.dropout_rate)(
            attn_out, deterministic=deterministic)

        x = hidden_states + attn_out

        mlp_ln = nn.LayerNorm(
            epsilon=cfg.layer_norm_epsilon,
            param_dtype=cfg.param_dtype,
            dtype=cfg.dtype
        )(x)
        mlp_out = MLP(cfg)(mlp_ln, deterministic=deterministic)

        # Residual
        x = x + mlp_out

        return x


class GPT2(nn.Module):
    config: GPT2Config

    def setup(self):
        # Embedding shared by input and output
        self.wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(
                self.config.initializer_range),
            param_dtype=self.config.param_dtype,
            dtype=self.config.dtype
        )
        self.wpe = self.param(
            "wpe",
            nn.initializers.normal(self.config.initializer_range),
            (self.config.max_position_embeddings, self.config.n_embd),
            dtype=self.config.param_dtype,
        )
        self.blocks = [TransformerBlock(self.config, name=f"transformer_block_{i}")
                       for i in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon,
            param_dtype=self.config.param_dtype,
            dtype=self.config.dtype
        )
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

    def __call__(self, input_ids, deterministic=True):
        # Input embedding
        position_ids = jnp.arange(input_ids.shape[1])[None, :]
        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe[position_ids, :].astype(self.config.dtype)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(
            hidden_states, deterministic=deterministic)

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, deterministic=deterministic)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Output logits by tying embeddings
        # We transpose the embedding matrix
        wte_tied = self.wte.embedding.T.astype(
            self.config.dtype
        )  # from [vocab_size, n_embd] to [n_embd, vocab_size]
        logits = jnp.einsum("bld,dk->blk", hidden_states, wte_tied)
        return logits
