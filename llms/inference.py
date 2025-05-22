import argparse

import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import optax
from flax.training import checkpoints
from flax.training.train_state import TrainState
import flax.jax_utils as flax_utils

from llms.models.utils import load_model
from llms.models.gpt2 import GPT2, GPT2Config

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def unshard_params(state):
    unsharded_params = jax.tree.map(
        lambda x: x[0] if x.shape[0] == 8 else x, state.params)
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
    generated = input_ids  # shape: (batch, cur_len)

    for _ in range(num_tokens_to_generate):
        # run the model on everything so far
        logits = state.apply_fn(
            {"params": state.params},
            generated,
            deterministic=True,
        )  # (batch, cur_len, vocab_size)

        # sample from the last position
        last_logits = logits[:, -1, :] / temperature
        rng, subrng = jax.random.split(rng)
        next_token = jax.random.categorical(subrng, last_logits)
        next_token = next_token[:, None]  # (batch,1)

        generated = jnp.concatenate([generated, next_token], axis=1)
    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Program to inference from GPT-2 model checkpoint")
    parser.add_argument('--path', type=str,
                        help="Path to downloaded model checkpoint")
    args = parser.parse_args()

    # ===================
    # Defining Model
    # ===================
    model = GPT2(GPT2Config())

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 1024), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']

    optimizer = optax.adamw(learning_rate=5e-5)
    state = TrainState.create(apply_fn=model.apply,
                              params=params, tx=optimizer)

    print("Available Devices:", jax.device_count())
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print("Total Parameters:", param_count, f"({param_count / 1e6:.2f}M)")

    state = checkpoints.restore_checkpoint(
        ckpt_dir=str(args.path),
        target=state,
        prefix="gpt_2_checkpoint_"
    )

    input_text = "Once upon a time"
    encoded_input = tokenizer.encode(
        input_text, return_tensors="np")  # shape: (1, prefix_length)
    input_ids = jnp.array(encoded_input)

    rng = jax.random.PRNGKey(0)
    num_tokens = 50
    temperature = 0.7

    state = flax_utils.unreplicate(state)

    final_ids = generate(
        state=state,
        input_ids=input_ids,
        rng=rng,
        num_tokens_to_generate=num_tokens,
        temperature=temperature,
    )

    decoded_text = tokenizer.decode(np.array(final_ids[0]))
    print(decoded_text)
