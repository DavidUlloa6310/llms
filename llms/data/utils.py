import pickle

import numpy as np
import jax
import jax.numpy as jnp


def batch_dataset(dataset, batch_size):
    batch = {"input_ids": []}
    for example in dataset:
        batch["input_ids"].append(example["input_ids"])
        if len(batch["input_ids"]) == batch_size:
            yield {k: np.array(v, dtype=np.int32) for k, v in batch.items()}
            batch = {"input_ids": []}

    if batch["input_ids"]:
        yield {k: np.array(v, dtype=np.int32) for k, v in batch.items()}


def shard_batch(batch):
    batch_size = batch["input_ids"].shape[0]
    per_device_batch = batch_size // jax.device_count()
    return jax.tree.map(lambda x: x.reshape((jax.device_count(), per_device_batch, -1)), batch)
