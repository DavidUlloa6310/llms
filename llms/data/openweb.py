from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import argparse


def build_openweb(split: str = "train", seq_len: int = 1024) -> DatasetDict:
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    ds = load_dataset("openwebtext", split=split)

    def tokenize_fn(examples):
        tokens = tok(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )
        return {"input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]}

    ds = ds.map(tokenize_fn,
                batched=True,
                remove_columns=["text"],
                num_proc=4)
    ds = ds.shuffle(seed=42)
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="Directory in which to save the processed dataset")
    args = parser.parse_args()

    out_dir = Path(args.path).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_openweb()
    # Arrow-based, reload with load_from_disk
    dataset.save_to_disk(out_dir)
    print(f"Saved OpenWebText to {out_dir}")
