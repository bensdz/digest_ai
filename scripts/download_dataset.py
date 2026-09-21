import os
import json
import torch
from datasets import load_dataset
from pathlib import Path
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
MODEL_DIR = PROJECT_ROOT / "data" / "tokenizer"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


print(" Loading dataset from Hugging Face via Parquet fallback...")
dataset = load_dataset(
    "parquet",
    data_files={
        "train": "hf://datasets/allenai/scitldr@refs/convert/parquet/Abstract/train/0000.parquet",
        "validation": "hf://datasets/allenai/scitldr@refs/convert/parquet/Abstract/validation/0000.parquet",
        "test": "hf://datasets/allenai/scitldr@refs/convert/parquet/Abstract/test/0000.parquet"
    }
)

print(" Saving raw text data to 'data/raw'...")
splits = ["train", "validation", "test"]


for split in splits:
    raw_file_path = RAW_DIR / f"{split}.jsonl"
    with open(raw_file_path, "w", encoding="utf-8") as f:
        for item in dataset[split]:
            # Flatten lists of strings into continuous text lines
            source_text = " ".join(item["source"])
            target_text = " ".join(item["target"])

            # Combine them as a plain document line or structure them
            line_data = {"source": source_text, "target": target_text}
            f.write(json.dumps(line_data) + "\n")

print(" Training custom BPE Tokenizer...")
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=50000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
)


raw_files = [str(RAW_DIR / f"{split}.jsonl") for split in splits]
tokenizer.train(raw_files, trainer)


tokenizer_path = MODEL_DIR / "digest_bpe_tokenizer.json"
tokenizer.save(str(tokenizer_path))
print(f" Tokenizer trained and saved to: {tokenizer_path}")


print(" Processing data into PyTorch Tensors...")
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
tokenizer.enable_truncation(max_length=512)


for split in splits:
    input_ids_list = []
    attention_masks_list = []

    raw_file_path = RAW_DIR / f"{split}.jsonl"
    with open(raw_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Create a single textual string
            full_text = f"[BOS] {data['source']} [SEP] {data['target']} [EOS]"

            # Encode text through our BPE tokenizer
            encoded = tokenizer.encode(full_text)

            input_ids_list.append(encoded.ids)
            attention_masks_list.append(encoded.attention_mask)

    # Convert into PyTorch Tensors
    processed_data = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks_list, dtype=torch.long)
    }

    # Save the PyTorch tensors to disk
    processed_file_path = PROCESSED_DIR / f"{split}_tensors.pt"
    torch.save(processed_data, processed_file_path)
    print(f" Saved packed tensors to: {processed_file_path}")

print("\n Data is organized and tokenized.")
