import argparse
import math
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.jax_utils as flax_utils
from flax.training import checkpoints
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm
import wandb
from datasets import load_from_disk

from llms.models.gpt2 import GPT2Config, GPT2
from llms.data.openweb import build_openweb
from llms.data.utils import shard_batch, batch_dataset


@partial(jax.pmap,
         axis_name="batch", static_broadcasted_argnums=(3,))
def train_step(state, batch, rng, clip_norm):
    dropout_key = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            deterministic=False,
            rngs={"dropout": dropout_key}
        )
        # jax.debug.print("logits in loss_fn: {x}", x=logits)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :], batch['input_ids'][:, 1:]
        ))

        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    # Needed for distributed training
    averaged_grads = jax.lax.pmean(grads, axis_name='batch')
    averaged_loss = jax.lax.pmean(loss, axis_name='batch')

    raw_norm = optax.global_norm(averaged_grads)

    # emulate what Adam receives after grad clipping
    clipper = optax.clip_by_global_norm(clip_norm)
    clipped_grads, _ = clipper.update(averaged_grads, optax.EmptyState(), None)
    used_norm = optax.global_norm(clipped_grads)

    new_state = state.apply_gradients(grads=averaged_grads)

    metrics = {
        "train_loss": averaged_loss,
        "grad_norm": raw_norm,
        "clipped_grad_norm": used_norm
    }

    return new_state, metrics


def build_scheduler(total_steps, config):
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=config.peak_lr,
        transition_steps=config.warmup_steps,
    )
    plateau_schedule = optax.constant_schedule(value=config.peak_lr)
    decay_steps = total_steps - config.warmup_steps - config.plateau_steps
    decay_schedule = optax.cosine_decay_schedule(
        init_value=config.peak_lr,
        decay_steps=decay_steps,
        alpha=(config.peak_lr * config.end_lr_factor) / config.peak_lr,
    )
    return optax.join_schedules(
        schedules=[warmup_schedule, plateau_schedule, decay_schedule],
        boundaries=[config.warmup_steps,
                    config.warmup_steps + config.plateau_steps],
    )


def main():
    parser = argparse.ArgumentParser(
        description="JAX/Flax GPT-2 pre-training script")
    parser.add_argument("--checkpoint-path", type=Path, required=True,
                        help="Directory to store msgpack checkpoints")
    parser.add_argument("--resume", action="store_true", required=False,
                        help="Resume training from latest checkpoint inside --resume-path folder")
    parser.add_argument("--dataset-path", type=Path, required=False,
                        help="OpenWebText dataset path (downloaded if absent)")
    parser.add_argument("--num-chkpts", type=int, default=10,
                        help="Number of checkpoints to save during training")
    args = parser.parse_args()
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(
        args.dataset_path) if args.dataset_path else build_openweb()

    model = GPT2(GPT2Config())

    wandb.login()
    wandb.init(
        project="gpt-2-pretraining",
        config=dict(
            epochs=1,
            seq_len=1024,
            batch_size=8 * jax.device_count(),
            accum_steps=4,
            max_norm=2.0,
            peak_lr=5e-5,
            warmup_steps=2_000,
            adam_eps=1e-8,
            adam_b1=0.9,
            adam_b2=0.95,
            weight_decay=0.01,
            end_lr_factor=0.1,
            plateau_steps=2_000,
        ),
    )
    wandb.define_metric("examples_seen")
    wandb.define_metric("train_loss", step_metric="examples_seen")
    wandb.define_metric("perplexity", step_metric="examples_seen")
    wandb.define_metric("grad_norm", step_metric="examples_seen")
    wandb.define_metric("clipped_grad_norm", step_metric="examples_seen")
    wandb.define_metric("learning_rate")
    config = wandb.config

    steps_per_epoch = math.ceil(len(dataset) / config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    lr_schedule = build_scheduler(total_steps, config)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_norm),
        optax.MultiSteps(
            optax.adamw(
                learning_rate=lr_schedule,
                b1=config.adam_b1,
                b2=config.adam_b2,
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            ),
            every_k_schedule=config.accum_steps,
        )
    )

    rng = jax.random.PRNGKey(0)
    init_rng, training_rng = jax.random.split(rng)

    dummy_in = jnp.zeros((1, config.seq_len), dtype=jnp.int32)
    params = model.init(init_rng, dummy_in)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if args.resume:
        state = checkpoints.restore_checkpoint(
            ckpt_dir=str(args.checkpoint_path),
            target=state,
            prefix="gpt_2_checkpoint_"
        )
        print("Restored model from checkpoint, step:", state.step)

    state = flax_utils.replicate(state)

    print(f"Available devices : {jax.device_count()}")
    param_cnt = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters  : {param_cnt:,} ({param_cnt/1e6:.2f} M)")

    print("Model dtype: ", state.params['blocks_0']
          ['MHSelfAttention_0']['Dense_0']['kernel'].dtype)

    # for checkpoint saving
    save_interval = max(1, total_steps // args.num_chkpts)
    print(f"Total Steps: {total_steps}, saving every {save_interval} steps")

    for epoch in range(1, config.epochs + 1):
        progress_bar = tqdm(desc=f"Epoch {epoch}", unit="batch")

        batches = (shard_batch(b) for b in batch_dataset(
            dataset, batch_size=config.batch_size))
        # Load only 3 batches into memory ahead of time (good for GPUs)
        prefetch_iter = flax_utils.prefetch_to_device(batches, size=3)

        for batch in prefetch_iter:
            # curr_step_rng split uniquely for each device
            curr_step_rng, training_rng = jax.random.split(training_rng)
            device_rngs = jax.random.split(curr_step_rng, jax.device_count())

            state, metrics = train_step(
                state, batch, device_rngs, config.max_norm)

            update_step = int(flax_utils.unreplicate(state).step)
            if (update_step > 0 and update_step % save_interval == 0) or (update_step == total_steps):
                chkpt_state = flax_utils.unreplicate(state)
                checkpoints.save_checkpoint(
                    args.checkpoint_path,
                    target=chkpt_state,
                    step=update_step,
                    overwrite=False,
                    keep=args.num_chkpts,
                    prefix="gpt_2_checkpoint_"
                )
                print(f"Saved model checkpoint at step {update_step}")

            if update_step % 100 == 0:
                progress_bar.set_postfix(loss=metrics['train_loss'][0])
            progress_bar.update(1)

            wandb.log({
                "examples_seen": update_step * config.batch_size * config.accum_steps,
                "train_loss": metrics['train_loss'][0],
                "perplexity": math.exp(metrics['train_loss'][0]),
                "grad_norm": metrics['grad_norm'][0],
                "clipped_grad_norm": metrics['clipped_grad_norm'][0],
                "learning_rate": lr_schedule(update_step),
            }, step=update_step, commit=update_step % 100 == 0)  # only commit every 100 steps

        print(f"Epoch {epoch} completed. Loss: {metrics['train_loss'][0]}")

    progress_bar.close()
    wandb.finish()


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "highest")
    main()
