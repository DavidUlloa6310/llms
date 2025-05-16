import argparse

import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import optax
from flax.training.train_state import TrainState
import flax.jax_utils as flax_utils

from llms.models.utils import load_model
from llms.models.gpt2 import GPT2, GPT2Config

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def unshard_params(state):
    unsharded_params = jax.tree.map(lambda x: x[0] if x.shape[0] == 8 else x, state.params)
    unsharded_state = TrainState.create(
        apply_fn=model.apply,
        params=unsharded_params,
        tx=optax.adamw(learning_rate=5e-5)
    )
    return unsharded_params, unsharded_state

def generate(
    state,
    input_ids,
    rng,
    config,
    num_tokens_to_generate=50,
    temperature=1.0,
):
    # logits, key_cache, value_cache = state.apply_fn(
    #     {"params": state.params},
    #     input_ids,
    #     # key_cache = None,
    #     # value_cache = None,
    #     deterministic=True
    # )
    logits = state.apply_fn(
        {"params": state.params},
        input_ids,
        # key_cache = key_cache,
        # value_cache = value_cache,
        deterministic=True
    )
    generated_ids = input_ids
    for i in range(num_tokens_to_generate):
        last_logits = logits[:, -1, :] / temperature
        
        rng, subrng = jax.random.split(rng)
        next_token_id = jax.random.categorical(subrng, last_logits)  # shape: (batch,)
        
        next_token_id = next_token_id[:, None]  # (batch, 1)
        generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)
        
        # logits_new, key_cache, value_cache = state.apply_fn(
        #     {"params": state.params},
        #     next_token_id,
        #     key_cache = key_cache,
        #     value_cache = value_cache,
        #     deterministic=True
        # )
        logits_new= state.apply_fn(
            {"params": state.params},
            next_token_id,
            # key_cache = key_cache,
            # value_cache = value_cache,
            deterministic=True
        )
        # logits_new shape: (batch, 1, vocab_size)
        logits = jnp.concatenate([logits, logits_new], axis=1)

    return generated_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Program to inference from GPT-2 model checkpoint")
    parser.add_argument('--path', type=str,
                        help="Path to downloaded model checkpoint")
    args = parser.parse_args()

    # ===================
    # Defining Model
    # ===================
    config = GPT2Config()
    model = GPT2(config)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 1024), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']

    optimizer = optax.adamw(learning_rate=5e-5)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    print("Available Devices:", jax.device_count())
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print("Total Parameters:", param_count, f"({param_count / 1e6:.2f}M)")

    state = load_model(state, args.path)

    input_text = "Once upon a time, Urne, a father"
    encoded_input = tokenizer.encode(input_text, return_tensors="np")  # shape: (1, prefix_length)
    input_ids = jnp.array(encoded_input)

    rng = jax.random.PRNGKey(0)
    num_tokens = 50
    temperature = 0.7

    _, unshareded_state = unshard_params(state)

    final_ids = generate(
        state = unshareded_state,
        input_ids=input_ids,
        rng=rng,
        config=config,
        num_tokens_to_generate=num_tokens,
        temperature=temperature,
    )

    decoded_text = tokenizer.decode(np.array(final_ids[0]))
    print(decoded_text)