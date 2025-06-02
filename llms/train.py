import logging
import argparse
import math
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils
import flax.nnx as nnx
import flax.jax_utils as flax_utils
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm
import wandb
from datasets import load_from_disk

from llms.models.gpt2 import GPT2
from llms.data.openweb import build_openweb
from llms.data.utils import batch_dataset


def train_step(model: GPT2, optimizer: nnx.Optimizer, batch, clip_norm):
    def loss_fn(model):
        logits = model(batch['input_ids'])
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :], batch['input_ids'][:, 1:]
        ))
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    raw_norm = optax.global_norm(grads)

    # emulate what Adam receives after grad clipping
    clipper = optax.clip_by_global_norm(clip_norm)
    clipped_grads, _ = clipper.update(grads, optax.EmptyState(), None)
    used_norm = optax.global_norm(clipped_grads)

    optimizer.update(grads)

    metrics = {
        "train_loss": loss,
        "grad_norm": raw_norm,
        "clipped_grad_norm": used_norm
    }

    return metrics


# JIT w/ sharding - not using decorator with args for caching purposes
mesh = Mesh(mesh_utils.create_device_mesh(
    (jax.device_count(), )), axis_names=("data", ))
replicate = NamedSharding(mesh, P())
batch_spec = NamedSharding(mesh, P('data', None))
with mesh:
    jit_train_step = nnx.jit(
        train_step,
        in_shardings=(replicate, replicate, batch_spec, None),
        out_shardings=replicate,
        # static_argnums=(3,)
    )


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


def build_model(resume: bool, checkpoint_path: str) -> Union[GPT2, int]:
    rngs = nnx.Rngs(0, params=1, dropout=2)
    if resume:
        with ocp.CheckpointManager(checkpoint_path, options=ocp.CheckpointManagerOptions()) as mngr:
            latest_step = mngr.latest_step()
            if latest_step is None:
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_path}")

            abs_model = nnx.eval_shape(lambda: GPT2(rngs))
            graph_def, abs_state = nnx.split(abs_model)
            # If loading sharded model, Orbax API expects PyTree of jax.SharedDtypeStruct
            # (https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html#load-a-sharded-model-from-a-checkpoint)
            # In our case, our model isn't sharded - it's copied over each device.
            state = mngr.restore(
                latest_step, args=ocp.StandardRestore(abs_state))
            return nnx.merge(graph_def, state), latest_step

    return GPT2(rngs), 0


def train(config, args):
    dataset = load_from_disk(
        args.dataset_path) if args.dataset_path else build_openweb()
    dataset = dataset.select(range(int(len(dataset) * args.dataset_slice)))

    logging.info("Available devices : %d", jax.device_count())
    model, prev_steps = build_model(args.resume, args.checkpoint_path)
    model = jax.device_put(model, replicate)

    param_cnt = sum(
        p.size for p in jax.tree_util.tree_leaves(nnx.state(model)))
    logging.info(
        "Total parameters: %d, (%.2f M)", param_cnt, param_cnt/1e6)

    total_steps = math.ceil(len(dataset) / config.batch_size)
    lr_schedule = build_scheduler(total_steps, config)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        ),
    )
    # optimizer updates happen in-place by carrying an internal reference to our model
    # if you modify our reference to model, optimizer will not update those parameters
    # so model must equal optimizer.model
    optimizer = nnx.Optimizer(model, tx)

    save_interval = max(1, total_steps // args.num_chkpts)
    logging.info("Total steps: %s, saving every %s steps",
                 total_steps, save_interval)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=args.num_chkpts,
        save_interval_steps=save_interval
    )

    batches = batch_dataset(dataset, batch_size=config.batch_size)
    with ocp.CheckpointManager(args.checkpoint_path, options=options) as mngr:
        progress_bar = tqdm(desc=f"Step {prev_steps}", unit="batch")
        for i, batch in enumerate(batches):
            step = prev_steps + i + 1

            metrics = jit_train_step(model, optimizer, batch, config.max_norm)

            # CheckpointManager considers our step_interval on every .save the save_interval
            mngr.save(step, args=ocp.args.StandardSave(
                nnx.state(model)))

            progress_bar.set_postfix(loss=metrics['train_loss'])
            progress_bar.update(1)

            wandb.log({
                "examples_seen": step * config.batch_size * config.accum_steps,
                "train_loss": metrics['train_loss'],
                "perplexity": math.exp(metrics['train_loss']),
                "grad_norm": metrics['grad_norm'],
                "clipped_grad_norm": metrics['clipped_grad_norm'],
                "learning_rate": lr_schedule(step),
            }, step=step, commit=step % 100 == 0)  # only commit every 100 steps

    logging.info("Training completed. Loss: %.4f", metrics['train_loss'])
    progress_bar.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True)

    parser = argparse.ArgumentParser(
        description="JAX/Flax GPT-2 pre-training script")
    parser.add_argument("--checkpoint-path", type=Path, required=True,
                        help="Directory to store msgpack checkpoints")
    parser.add_argument("--resume", action="store_true", required=False,
                        help="Resume training from latest checkpoint inside --resume-path folder")
    parser.add_argument("--dataset-path", type=Path, required=False,
                        help="OpenWebText dataset path (downloaded if absent)")
    parser.add_argument("--dataset-slice", type=float, required=False, default=0.1,
                        help="Amount of dataset GPT-2 should be trained on (default = 0.5)")
    parser.add_argument("--num-chkpts", type=int, default=10,
                        help="Number of checkpoints to save during training (default = 10)")
    args = parser.parse_args()
    args.checkpoint_path.mkdir(parents=True, exist_ok=True)

    wandb.login()
    wandb.init(
        project="gpt-2-pretraining",
        config=dict(
            epochs=1,
            seq_len=1024,
            batch_size=jax.device_count(),
            accum_steps=4,
            max_norm=2.0,
            peak_lr=5e-4,
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

    try:
        train(wandb.config, args)
    finally:
        wandb.finish()
