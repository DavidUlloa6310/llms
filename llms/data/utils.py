import pickle

import jax
import jax.numpy as jnp

def save_dataset(dataset, path: str, filename: str):
    with open(f"{path}/{filename}", "wb") as f:
        pickle.dump(dataset, f)


def read_dataset(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def batch_dataset(dataset, batch_size):
    batch = {"input_ids": []}
    for example in dataset:
        batch["input_ids"].append(example["input_ids"])
        if len(batch["input_ids"]) == batch_size:
            yield {k: jnp.array(v, dtype=jnp.int32) for k, v in batch.items()}
            batch = {"input_ids": []}


def shard_batch(batch):
    batch_size = batch["input_ids"].shape[0]
    per_device_batch = batch_size // jax.device_count()
    return jax.tree.map(lambda x: x.reshape((jax.device_count(), per_device_batch, -1)), batch)
