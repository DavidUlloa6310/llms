import argparse
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer
import pickle
import jax
import jax.numpy as jnp


tokenizer = AutoTokenizer.from_pretrained("gpt2")


def download_openweb():
    dataset = load_dataset(
        "openwebtext"
    )

    def tokenize_function(examples):
        return {
            "input_ids": tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=1024
            )["input_ids"]
        }

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.shuffle(seed=42)

    return dataset


def save_dataset(dataset, path: str):
    with open(path, "wb") as f:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to download and store openwebtext")
    parser.add_argument('--path', type=str,
                        help="Path to downlaod openwebtext")
    args = parser.parse_args()

    dataset = download_openweb()
    save_dataset(dataset, args.path)
    print(f"Saved openwebtext dataset to '{args.path}'")
