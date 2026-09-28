import os
import json
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from huggingface_hub import login


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TOKENIZER_DIR = PROJECT_ROOT / "data" / "tokenizer"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


SOURCE_MAX_LENGTH = 512
TARGET_MAX_LENGTH = 128

VOCAB_SIZE = 16000

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
]

SPLITS = [
    "train",
    "validation",
    "test",
]


print("Loading dataset from Hugging Face...")

dataset = load_dataset(
    "parquet",
    data_files={
        "train": (
            "hf://datasets/allenai/scitldr"
            "@refs/convert/parquet/Abstract/train/0000.parquet"
        ),
        "validation": (
            "hf://datasets/allenai/scitldr"
            "@refs/convert/parquet/Abstract/validation/0000.parquet"
        ),
        "test": (
            "hf://datasets/allenai/scitldr"
            "@refs/convert/parquet/Abstract/test/0000.parquet"
        ),
    },
)


print("\nExample from training dataset:")
print(dataset["train"][0])

# Save raw data as JSONL

print("\nSaving raw text data...")

for split in SPLITS:

    raw_file_path = RAW_DIR / f"{split}.jsonl"

    with open(raw_file_path, "w", encoding="utf-8") as f:

        for item in dataset[split]:

            source = item["source"]
            target = item["target"]

            if isinstance(source, list):
                source_text = " ".join(source)
            else:
                source_text = source

            if isinstance(target, list):
                target_text = " ".join(target)
            else:
                target_text = target

            example = {
                "source": source_text,
                "target": target_text,
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved: {raw_file_path}")


# Train BPE tokenizer

print("\nTraining BPE tokenizer...")

tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = Whitespace()


trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
)


train_file = str(RAW_DIR / "train.jsonl")

tokenizer.train(
    [train_file],
    trainer,
)


# Get special token IDs

pad_id = tokenizer.token_to_id("[PAD]")
bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")
unk_id = tokenizer.token_to_id("[UNK]")


print("\nSpecial token IDs:")
print(f"[PAD] = {pad_id}")
print(f"[BOS] = {bos_id}")
print(f"[EOS] = {eos_id}")
print(f"[UNK] = {unk_id}")


tokenizer_path = TOKENIZER_DIR / "digest_bpe_tokenizer.json"

tokenizer.save(str(tokenizer_path))

print(f"\nTokenizer saved to:")
print(tokenizer_path)


print("\nTokenizing dataset...")


def encode_source(text):
    # Encode a source document.

    encoded = tokenizer.encode(text)

    ids = [bos_id] + encoded.ids + [eos_id]

    # Truncate while keeping EOS.
    ids = ids[:SOURCE_MAX_LENGTH]

    if ids[-1] != eos_id:
        ids[-1] = eos_id

    attention_mask = [1] * len(ids)

    # Padding
    padding_length = SOURCE_MAX_LENGTH - len(ids)

    ids += [pad_id] * padding_length
    attention_mask += [0] * padding_length

    return ids, attention_mask


def encode_target(text):
    # Encode a target summary.

    encoded = tokenizer.encode(text)

    ids = [bos_id] + encoded.ids + [eos_id]

    # Truncate while keeping EOS.
    ids = ids[:TARGET_MAX_LENGTH]

    if ids[-1] != eos_id:
        ids[-1] = eos_id

    attention_mask = [1] * len(ids)

    # Padding
    padding_length = TARGET_MAX_LENGTH - len(ids)

    ids += [pad_id] * padding_length
    attention_mask += [0] * padding_length

    return ids, attention_mask


for split in SPLITS:

    print(f"\nProcessing {split} split...")

    source_ids_list = []
    source_attention_masks = []

    target_ids_list = []
    target_attention_masks = []

    raw_file_path = RAW_DIR / f"{split}.jsonl"

    with open(raw_file_path, "r", encoding="utf-8") as f:

        for line in f:

            data = json.loads(line)

            source_text = data["source"]
            target_text = data["target"]

            source_ids, source_mask = encode_source(
                source_text
            )

            source_ids_list.append(source_ids)
            source_attention_masks.append(source_mask)

            target_ids, target_mask = encode_target(
                target_text
            )

            target_ids_list.append(target_ids)
            target_attention_masks.append(target_mask)

    processed_data = {

        "source_ids": torch.tensor(
            source_ids_list,
            dtype=torch.long,
        ),

        "source_attention_mask": torch.tensor(
            source_attention_masks,
            dtype=torch.long,
        ),

        "target_ids": torch.tensor(
            target_ids_list,
            dtype=torch.long,
        ),

        "target_attention_mask": torch.tensor(
            target_attention_masks,
            dtype=torch.long,
        ),
    }

    processed_file_path = (
        PROCESSED_DIR / f"{split}_tensors.pt"
    )

    torch.save(
        processed_data,
        processed_file_path,
    )

    print(f"Saved: {processed_file_path}")

    print(
        f"  source_ids: "
        f"{processed_data['source_ids'].shape}"
    )

    print(
        f"  target_ids: "
        f"{processed_data['target_ids'].shape}"
    )


print("\n========================================")
print("Data preprocessing complete!")
print("========================================")

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print(f"Source max length: {SOURCE_MAX_LENGTH}")
print(f"Target max length: {TARGET_MAX_LENGTH}")

print("\nOutput structure:")
print("data/")
print("├── raw/")
print("│   ├── train.jsonl")
print("│   ├── validation.jsonl")
print("│   └── test.jsonl")
print("│")
print("├── tokenizer/")
print("│   └── digest_bpe_tokenizer.json")
print("│")
print("└── processed/")
print("    ├── train_tensors.pt")
print("    ├── validation_tensors.pt")
print("    └── test_tensors.pt")
