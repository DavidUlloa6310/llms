import argparse
import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.jax_utils as flax_utils
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm
import wandb

from llms.models.gpt2 import GPT2Config, GPT2
from llms.models.utils import save_model
from llms.data.openweb import download_openweb
from llms.data.utils import shard_batch, batch_dataset, read_dataset


@partial(jax.pmap,
         axis_name="batch",
         static_broadcasted_argnums=(3,))
def train_step(state, batch, dropout_rngs, max_grad_norm):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            deterministic=False,
            rngs={"dropout": dropout_rngs}
        )
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :], batch['input_ids'][:, 1:]
        ))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Needed for distributed training
    averaged_grads = jax.lax.pmean(grads, axis_name='batch')
    averaged_loss = jax.lax.pmean(loss, axis_name='batch')

    clipper = optax.clip_by_global_norm(max_grad_norm)
    clipped_grads, _ = clipper.update(averaged_grads, optax.EmptyState(), None)

    new_state = state.apply_gradients(grads=clipped_grads)
    grad_norm = optax.global_norm(averaged_grads)
    return new_state, averaged_loss, grad_norm


def main():
    parser = argparse.ArgumentParser(
        description="JAX/Flax GPT-2 pre-training script")
    parser.add_argument("--checkpoint-path", type=Path, required=True,
                        help="Directory to store msgpack checkpoints")
    parser.add_argument("--dataset-path", type=Path,
                        help="OpenWebText dataset path (downloaded if absent)")
    args = parser.parse_args()
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)

    dataset = (read_dataset(args.dataset_path) if args.dataset_path
               else download_openweb())["train"]

    config = GPT2Config()
    model = GPT2(config)

    wandb.login()
    run = wandb.init(
        project="gpt-2-pretraining",
        config=dict(
            epochs=1,
            seq_len=1024,
            batch_size=jax.device_count(),
            accum_steps=8,
            max_norm=2.0,
            peak_lr=1e-6,
            warmup_steps=2_000,
            adam_eps=1e-8,
            adam_b1=0.9,
            adam_b2=0.95,
            weight_decay=0.01,
            end_lr_factor=0.1,
        ),
    )
    config = wandb.config

    steps_per_epoch = math.ceil(len(dataset) / jax.device_count())
    total_steps = config.epochs * steps_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=total_steps - config.warmup_steps,
        end_value=config.peak_lr * config.end_lr_factor,
    )
    tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    rng = jax.random.PRNGKey(0)
    init_rng, training_rng = jax.random.split(rng)
    dummy_in = jnp.zeros((1, config.seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_in)["params"]

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = flax_utils.replicate(state)

    param_cnt = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Available devices : {jax.device_count()}")
    print(f"Total parameters  : {param_cnt:,} ({param_cnt/1e6:.2f} M)")

    for epoch in range(1, config.epochs + 1):
        step = 0
        progress_bar = tqdm(desc=f"Epoch {epoch}", unit="batch")

        for batch in batch_dataset(dataset, batch_size=config.batch_size):
            batch = shard_batch(batch)

            curr_step_rng, training_rng = jax.random.split(training_rng)
            # curr_step_rng split uniquely for each device
            dropout_rngs = jax.random.split(curr_step_rng, jax.device_count())

            state, loss, grad_norm = train_step(
                state, batch, dropout_rngs, config.max_norm)

            if step % 100 == 0:
                progress_bar.set_postfix(loss=loss[0])
            step += 1
            progress_bar.update(1)

            wandb.log({
                "train_loss": loss[0],
                "grad_norm": grad_norm[0],
                "learning_rate": lr_schedule(step),
            }, step)

        save_model(state, f"checkpoint_epoch_{epoch}.msgpack")
        print(f"Epoch {epoch} completed. Loss: {loss.mean()}")

    progress_bar.close()
    wandb.finish()


if __name__ == "__main__":
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    main()
