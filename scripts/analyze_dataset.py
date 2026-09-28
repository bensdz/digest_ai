from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "tokenizer"
    / "digest_bpe_tokenizer.json"
)


# ============================================================
# 2. Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

print(
    f"Vocabulary size: "
    f"{tokenizer.get_vocab_size()}"
)


# ============================================================
# 3. Tokenize dataset and collect lengths
# ============================================================

def analyze_split(split):

    file_path = RAW_DIR / f"{split}.jsonl"

    source_lengths = []
    target_lengths = []

    print(f"\nAnalyzing {split}...")
    print(f"File: {file_path}")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            data = json.loads(line)

            source = data["source"]
            target = data["target"]

            source_encoding = tokenizer.encode(source)
            target_encoding = tokenizer.encode(target)

            source_lengths.append(
                len(source_encoding.ids) + 2
            )

            target_lengths.append(
                len(target_encoding.ids) + 2
            )

    return (
        np.array(source_lengths),
        np.array(target_lengths)
    )


# ============================================================
# 4. Statistics
# ============================================================

def print_statistics(name, lengths):

    percentiles = [
        50,
        75,
        90,
        95,
        99,
        99.5,
        100,
    ]

    print(f"\n{name}")
    print("-" * 50)

    print(f"Number of examples : {len(lengths)}")
    print(f"Minimum            : {lengths.min()}")
    print(f"Mean               : {lengths.mean():.1f}")
    print(f"Median             : {np.median(lengths):.1f}")
    print(f"Maximum            : {lengths.max()}")

    print("\nPercentiles:")

    for p in percentiles:

        value = np.percentile(lengths, p)

        print(
            f"{p:>5}% : "
            f"{value:.0f} tokens"
        )


# ============================================================
# 5. Calculate truncation percentage
# ============================================================

def truncation_rate(lengths, max_length):

    truncated = np.sum(
        lengths > max_length
    )

    percentage = (
        truncated / len(lengths)
    ) * 100

    return truncated, percentage


# ============================================================
# 6. Analyze all splits
# ============================================================

all_data = {}

for split in [
    "train",
    "validation",
    "test",
]:

    source_lengths, target_lengths = (
        analyze_split(split)
    )

    all_data[split] = {
        "source": source_lengths,
        "target": target_lengths,
    }

    print_statistics(
        f"{split.upper()} SOURCE",
        source_lengths,
    )

    print_statistics(
        f"{split.upper()} TARGET",
        target_lengths,
    )


# ============================================================
# 7. Combine splits
# ============================================================

all_source_lengths = np.concatenate(
    [
        all_data[split]["source"]
        for split in all_data
    ]
)

all_target_lengths = np.concatenate(
    [
        all_data[split]["target"]
        for split in all_data
    ]
)


# ============================================================
# 8. Final dataset statistics
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL DATASET STATISTICS")
print("=" * 70)

print_statistics(
    "SOURCE",
    all_source_lengths,
)

print_statistics(
    "TARGET",
    all_target_lengths,
)


# ============================================================
# 9. Test candidate maximum lengths
# ============================================================

print("\n")
print("=" * 70)
print("SOURCE LENGTH OPTIONS")
print("=" * 70)

source_candidates = [
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    4096,
]

for max_length in source_candidates:

    truncated, percentage = truncation_rate(
        all_source_lengths,
        max_length,
    )

    print(
        f"{max_length:>5} tokens : "
        f"{truncated:>5} examples truncated "
        f"({percentage:.2f}%)"
    )


print("\n")
print("=" * 70)
print("TARGET LENGTH OPTIONS")
print("=" * 70)

target_candidates = [
    64,
    96,
    128,
    160,
    192,
    256,
    384,
    512,
]

for max_length in target_candidates:

    truncated, percentage = truncation_rate(
        all_target_lengths,
        max_length,
    )

    print(
        f"{max_length:>5} tokens : "
        f"{truncated:>5} examples truncated "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 10. Histograms
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    all_source_lengths,
    bins=50,
)

plt.xlabel("Source token length")
plt.ylabel("Number of examples")
plt.title("Source Token Length Distribution")

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT
    / "source_length_distribution.png"
)

plt.show()


plt.figure(figsize=(10, 6))

plt.hist(
    all_target_lengths,
    bins=50,
)

plt.xlabel("Target token length")
plt.ylabel("Number of examples")
plt.title("Target Token Length Distribution")

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT
    / "target_length_distribution.png"
)

plt.show()


# ============================================================
# 11. Find longest examples
# ============================================================

print("\n")
print("=" * 70)
print("LONGEST SOURCE EXAMPLES")
print("=" * 70)

train_file = RAW_DIR / "train.jsonl"

examples = []

with open(
    train_file,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        data = json.loads(line)

        source = data["source"]

        length = len(
            tokenizer.encode(source).ids
        ) + 2

        examples.append(
            (
                length,
                source,
                data["target"],
            )
        )


examples.sort(
    key=lambda x: x[0],
    reverse=True,
)


for i, (length, source, target) in enumerate(
    examples[:10],
    start=1,
):

    print(f"\nExample {i}")
    print(f"Source tokens: {length}")
    print(f"Source preview: {source[:300]}...")
    print(f"Target preview: {target[:300]}...")


# ============================================================
# 12. Recommendation helper
# ============================================================

print("\n")
print("=" * 70)
print("POSSIBLE CONFIGURATION")
print("=" * 70)

source_p95 = np.percentile(
    all_source_lengths,
    95,
)

source_p99 = np.percentile(
    all_source_lengths,
    99,
)

target_p95 = np.percentile(
    all_target_lengths,
    95,
)

target_p99 = np.percentile(
    all_target_lengths,
    99,
)

print(
    f"Source P95 : {source_p95:.0f} tokens"
)

print(
    f"Source P99 : {source_p99:.0f} tokens"
)

print(
    f"Target P95 : {target_p95:.0f} tokens"
)

print(
    f"Target P99 : {target_p99:.0f} tokens"
)

print("\nUse these numbers to choose:")
print("SOURCE_MAX_LENGTH")
print("TARGET_MAX_LENGTH")

print("\nDo not automatically choose the maximum.")
print("Longer sequences require more GPU memory.")
