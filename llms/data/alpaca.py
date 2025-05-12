import argparse
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer
import pickle
import jax
import jax.numpy as jnp

from llms.data.utils import save_dataset


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def download_alpaca():
    dataset = load_dataset("tatsu-lab/alpaca")
    def tokenize_function(examples):
        return {
            "input_ids": tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=1024
            )["input_ids"]
        }

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.shuffle(seed=42)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to download and store openwebtext")
    parser.add_argument('--path', type=str,
                        help="Path to downlaod openwebtext")
    args = parser.parse_args()

    dataset = download_alpaca()
    save_dataset(dataset, args.path, "alpaca.pkl")
    print(f"Saved alpaca dataset to '{args.path}'")