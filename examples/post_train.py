import argparse
from dataclasses import dataclass


import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.jax_utils as flax_utils
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm


from llms.models.gpt2 import GPT2Config, GPT2
from llms.models.utils import save_model, load_model
from llms.data.alpaca import download_alpaca
from llms.data.utils import shard_batch, batch_dataset, read_dataset
from llms.train import TrainingParams, train_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Program to fine-tune a GPT-2 model as defined in models/gpt2.py")
    parser.add_argument('--model-checkpoint', type=str,
                        help="Path to model which will be fine-tuned")
    parser.add_argument('--checkpoint-path', type=str,
                        help="Path to download model checkpoints")
    parser.add_argument('--dataset-path', type=str, help = "Path to downloaded openweb dataset. If not provided, dataset is downloaded")
    parser.add_argument('--num-saves', type=int, help = "Number of checkpoints to save of model")
    args = parser.parse_args()


    config = GPT2Config()
    model = GPT2(config)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 1024), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']

    optimizer = optax.adamw(learning_rate=5e-5)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    load_model(state, parser.model_checkpoint)

    print("Available Devices:", jax.device_count())
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print("Total Parameters:", param_count, f"({param_count / 1e6:.2f}M)")

    # ===================
    # Training Model
    # ===================
    training = TrainingParams()
    rngs = jax.random.split(rng, jax.local_device_count())
    state = flax_utils.replicate(state)

    dataset = read_dataset(args.dataset_path) if args.dataset_path else download_alpaca()
    dataset = dataset['train']

    batches_per_save = (len(dataset) // training.batch_size) // args.num_saves

    for epoch in range(1, training.num_epochs + 1):
        step = 0
        progress_bar = tqdm(desc=f"Epoch {epoch}", unit="batch")

        for batch in batch_dataset(dataset, batch_size=training.batch_size):
            batch = shard_batch(batch)  # => (8, 1, 1024)
            state, loss = train_step(state, rngs, batch)

            if step % 100 == 0:
                progress_bar.set_postfix(loss=loss.mean().item())
            
            if step % batches_per_save == 0:
                save = 1
                save_model(state, f"{args.checkpoint_path}/alpaca_finetuning_{save}.msgpack")
                save += 1

            step += 1
            progress_bar.update(1)

        save_model(state, f"{args.checkpoint_path}/alpaca_finetuning_final.msgpack")

        print(f"Epoch {epoch} completed. Loss: {loss.mean()}")
        progress_bar.close()