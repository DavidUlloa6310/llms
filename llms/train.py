import argparse
from dataclasses import dataclass


import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.jax_utils as flax_utils
from flax.training.train_state import TrainState
import optax
import tqdm


from llms.models.gpt2 import GPT2Config, GPT2
from llms.models.utils import save_model
from llms.datasets.openweb import download_openweb, shard_batch, batch_dataset, save_model, read_dataset


@dataclass
class TrainingParams:
    num_epochs = 1
    batch_size = 8
    seq_length = 1024


@jax.pmap
def train_step(state, rng, batch):
    rng, dropout_rng = jax.random.split(rng)

    def loss_fn(params):

        logits = state.apply_fn(
            {'params': params},
            batch['input_ids'], 
            deterministic = False,
            rngs = {"dropout": dropout_rng}
        )

        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1, :], batch['input_ids'][:, 1:]
            )
        )
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Program to train GPT-2 model as defined in models/gpt2.py")
    parser.add_argument('--checkpoint-path', type=str,
                        help="Path to downlaod model checkpoints")
    parser.add_agument('--dataset-path', type=str, help = "Path to downloaded openweb dataset. If not provided, dataset is downloaded")
    parser.add_argument('--num-saves', type=int, help = "Number of checkpoints to save of model")
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

    # ===================
    # Training Model
    # ===================
    training = TrainingParams()
    rngs = jax.random.split(rng, jax.local_device_count())
    state = flax_utils.replicate(state)

    dataset = read_dataset(args.dataset_path) if args.dataset_path else download_openweb()

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
                save_model(state, f"{args.checkpoint_path}/gpt_2_checkpoint_{save}.msgpack")
                save += 1

            step += 1
            progress_bar.update(1)

        save_model(state, f"{args.checkpoint_path}/gpt_2_checkpoint_final.msgpack")

        print(f"Epoch {epoch} completed. Loss: {loss.mean()}")
        progress_bar.close()