from train import DigestTransformer, create_causal_mask
from tokenizers import Tokenizer
import torch
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKENIZER_PATH = (
    PROJECT_ROOT / "data" / "tokenizer" / "digest_bpe_tokenizer.json"
)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"

VOCAB_SIZE = 16000
SOURCE_MAX_LENGTH = 512
TARGET_MAX_LENGTH = 128

D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Load tokenizer and special-token IDs
# --------------------------------------------------

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

PAD_ID = tokenizer.token_to_id("[PAD]")
BOS_ID = tokenizer.token_to_id("[BOS]")
EOS_ID = tokenizer.token_to_id("[EOS]")

if None in (PAD_ID, BOS_ID, EOS_ID):
    raise ValueError("Tokenizer is missing a required special token.")


# --------------------------------------------------
# 3. Recreate the model and load trained weights
# --------------------------------------------------

model = DigestTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    max_source_length=SOURCE_MAX_LENGTH,
    max_target_length=TARGET_MAX_LENGTH,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from:", CHECKPOINT_PATH)
print("Device:", DEVICE)


# --------------------------------------------------
# 4. Convert source text into model input
# --------------------------------------------------

def encode_source(text):
    token_ids = tokenizer.encode(text).ids

    # Reserve space for BOS and EOS.
    token_ids = token_ids[: SOURCE_MAX_LENGTH - 2]
    token_ids = [BOS_ID] + token_ids + [EOS_ID]

    attention_mask = [1] * len(token_ids)

    padding_length = SOURCE_MAX_LENGTH - len(token_ids)

    token_ids += [PAD_ID] * padding_length
    attention_mask += [0] * padding_length

    source_ids = torch.tensor(
        [token_ids], dtype=torch.long, device=DEVICE
    )
    source_attention_mask = torch.tensor(
        [attention_mask], dtype=torch.long, device=DEVICE
    )

    return source_ids, source_attention_mask


# --------------------------------------------------
# 5. Generate a summary one token at a time
# --------------------------------------------------

@torch.no_grad()
def generate_summary(text):
    source_ids, source_attention_mask = encode_source(text)

    source_padding_mask = source_attention_mask == 0

    # Start generation with BOS.
    generated_ids = [BOS_ID]

    for _ in range(TARGET_MAX_LENGTH - 1):
        decoder_input = torch.tensor(
            [generated_ids], dtype=torch.long, device=DEVICE
        )

        # No padding exists in this generated prefix.
        target_padding_mask = torch.zeros_like(
            decoder_input, dtype=torch.bool
        )

        target_mask = create_causal_mask(
            decoder_input.size(1), DEVICE
        )

        logits = model(
            source_ids,
            decoder_input,
            source_padding_mask,
            target_padding_mask,
            target_mask,
        )

        # Get scores for the next token only.
        next_token_logits = logits[0, -1, :]

        # Greedy decoding: choose the highest-scoring token.
        next_token_id = next_token_logits.argmax().item()

        if next_token_id == EOS_ID:
            break

        generated_ids.append(next_token_id)

    # Remove BOS before converting IDs back into text.
    summary_ids = generated_ids[1:]

    return tokenizer.decode(summary_ids)


# --------------------------------------------------
# 6. Try Digest on your own text
# --------------------------------------------------

if __name__ == "__main__":
    abstract = input("Paste a scientific abstract:\n")

    summary = generate_summary(abstract)

    print("\nGenerated summary:")
    print(summary)
