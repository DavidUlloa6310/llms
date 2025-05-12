import flax.linen as nn
import jax.numpy as jnp

from llms.models.gpt2 import GPT2Config, MLP

class MHSelfAttention(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, hidden_states, key_cache, value_cache, deterministic: bool = True):
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
            bias_init=nn.initializers.zeros
        )(hidden_states)

        qkv = qkv.reshape(batch_size, seq_length, 3, cfg.n_head, cfg.n_embd // cfg.n_head)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(axis=2)  # [batch_size, seq_length, n_head, head_dim]
        k = k.squeeze(axis=2)
        v = v.squeeze(axis=2)

        # Transpose for (batch, head, seq, head_dim)
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        if key_cache is not None and value_cache is not None:
            k = jnp.concatenate([key_cache, k], axis = 2)
            v = jnp.concatenate([value_cache, v], axis = 2)

        full_seq_len = k.shape[2] # includes previously seen and new tokens
        new_seq_len = q.shape[2] # only tokens that already cached

        causal_mask = jnp.tril(jnp.ones((full_seq_len, full_seq_len)))
        if key_cache is not None and value_cache is not None:
            causal_mask = causal_mask[-new_seq_len:, :]

        causal_mask = causal_mask[None, None, :, :]

        dk = jnp.sqrt(k.shape[-1]).astype(q.dtype)

        # Full Attention Equation (w/ masking)
        attn_weights = jnp.matmul(q, k.transpose((0, 1, 3, 2))) / dk
        attn_weights = jnp.where(causal_mask == 0, -1e10, attn_weights)
        attn_probs = nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(rate=cfg.attn_dropout_rate)(attn_probs, deterministic=deterministic)
        attn_output = jnp.matmul(attn_probs, v)
        attn_output = attn_output.transpose((0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_length, cfg.n_embd)

        out = nn.Dense(
            cfg.n_embd,
            use_bias=True,
            kernel_init=nn.initializers.normal(cfg.initializer_range),
            bias_init=nn.initializers.zeros
        )(attn_output)

        return out, k, v

class TransformerBlock(nn.Module):
    config: GPT2Config

    @nn.compact
    def __call__(self, hidden_states, key_cache, value_cache, deterministic: bool = True):
        cfg = self.config

        attn_ln = nn.LayerNorm(epsilon=cfg.layer_norm_epsilon)(hidden_states)
        attn_out, new_k, new_v= MHSelfAttention(cfg)(attn_ln, key_cache, value_cache, deterministic=deterministic)
        attn_out = nn.Dropout(rate=cfg.dropout_rate)(attn_out, deterministic=deterministic)

        x = hidden_states + attn_out

        mlp_ln = nn.LayerNorm(epsilon=cfg.layer_norm_epsilon)(x)
        mlp_out = MLP(cfg)(mlp_ln, deterministic=deterministic)

        # Residual
        x = x + mlp_out

        return x, new_k, new_v

class GPT2(nn.Module):
    config: GPT2Config

    def setup(self):
        # Embedding shared by input and output
        self.wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(self.config.initializer_range)
        )
        self.wpe = self.param(
            "wpe",
            nn.initializers.normal(self.config.initializer_range),
            (self.config.max_position_embeddings, self.config.n_embd)
        )
        self.blocks = [TransformerBlock(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)

    def __call__(self, input_ids, key_cache = None, value_cache = None, deterministic = True):
        pos_offset = 0
        if key_cache is not None and value_cache is not None:
            pos_offset = key_cache[0].shape[2]

        seq_len = input_ids.shape[1]
        position_ids = jnp.arange(pos_offset, pos_offset + seq_len, dtype=jnp.int32)[None, :]
        position_embeds = self.wpe[position_ids, :]

        token_embeds = self.wte(input_ids)
        hidden_states = token_embeds + position_embeds

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        new_key_caches = []
        new_value_caches = []
        for i, block in enumerate(self.blocks):
            # For each block, pass the corresponding cache (if available).
            hidden_states, new_k, new_v = block(
                hidden_states, 
                deterministic=deterministic, 
                key_cache = key_cache[i, :, :] if key_cache is not None else None, 
                value_cache = value_cache[i, :, :] if key_cache is not None else None
            )
            new_key_caches.append(new_k)
            new_value_caches.append(new_v)

        hidden_states = self.ln_f(hidden_states)

        wte_tied = self.wte.embedding.T  # shape: [n_embd, vocab_size]
        logits = jnp.einsum("bld,dk->blk", hidden_states, wte_tied)
        return logits, new_key_caches, new_value_caches