import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Data
SOURCE_MAX_LENGTH = 512
TARGET_MAX_LENGTH = 128

# Model
D_MODEL = 256
NHEAD = 4

NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4

DIM_FEEDFORWARD = 1024

DROPOUT = 0.1

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 10

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

# Misc
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. Dataset
# ============================================================

class DigestDataset(Dataset):

    def __init__(self, file_path):

        data = torch.load(
            file_path,
            map_location="cpu",
        )

        self.source_ids = data["source_ids"]

        self.source_attention_mask = (
            data["source_attention_mask"]
        )

        self.target_ids = data["target_ids"]

        self.target_attention_mask = (
            data["target_attention_mask"]
        )

    def __len__(self):

        return len(self.source_ids)

    def __getitem__(self, index):

        return {
            "source_ids": self.source_ids[index],

            "source_attention_mask":
                self.source_attention_mask[index],

            "target_ids": self.target_ids[index],

            "target_attention_mask":
                self.target_attention_mask[index],
        }


# ============================================================
# 3. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_length,
    ):

        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_length,
            d_model,
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Shape:
        # [max_length, d_model]
        self.register_buffer(
            "pe",
            pe,
        )

    def forward(self, x):

        # x:
        # [batch, sequence_length, d_model]

        sequence_length = x.size(1)

        return x + self.pe[
            :sequence_length
        ].unsqueeze(0)


# ============================================================
# 4. Transformer Model
# ============================================================

class DigestTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        max_source_length,
        max_target_length,
    ):

        super().__init__()

        self.d_model = d_model

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.source_position = PositionalEncoding(
            d_model,
            max_source_length,
        )

        self.target_position = PositionalEncoding(
            d_model,
            max_target_length,
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.output = nn.Linear(
            d_model,
            vocab_size,
        )

    def forward(
        self,
        source_ids,
        target_ids,
        source_padding_mask,
        target_padding_mask,
        target_mask,
    ):

        # ====================================================
        # Encoder
        # ====================================================

        source = self.embedding(
            source_ids
        ) * math.sqrt(self.d_model)

        source = self.source_position(
            source
        )

        memory = self.encoder(
            source,
            src_key_padding_mask=source_padding_mask,
        )

        # ====================================================
        # Decoder
        # ====================================================

        target = self.embedding(
            target_ids
        ) * math.sqrt(self.d_model)

        target = self.target_position(
            target
        )

        output = self.decoder(
            target,
            memory,

            tgt_mask=target_mask,

            tgt_key_padding_mask=target_padding_mask,

            memory_key_padding_mask=source_padding_mask,
        )

        # ====================================================
        # Vocabulary logits
        # ====================================================

        logits = self.output(
            output
        )

        return logits


# ============================================================
# 5. Causal mask
# ============================================================

def create_causal_mask(
    sequence_length,
    device,
):

    mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            device=device,
        ),
        diagonal=1,
    )

    mask = mask.masked_fill(
        mask == 1,
        float("-inf"),
    )

    return mask


# ============================================================
# 6. Load datasets
# ============================================================

print("=" * 60)
print("Loading datasets")
print("=" * 60)

train_dataset = DigestDataset(
    PROCESSED_DIR / "train_tensors.pt"
)

validation_dataset = DigestDataset(
    PROCESSED_DIR / "validation_tensors.pt"
)

print(
    f"Train examples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation examples: "
    f"{len(validation_dataset)}"
)


# ============================================================
# 7. DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# 8. Load vocabulary size
# ============================================================

# The tokenizer has 16,000 tokens.
#
# For now we keep this explicit because the tokenizer is
# already configured with VOCAB_SIZE = 16000.

VOCAB_SIZE = 16000


# ============================================================
# 9. Create model
# ============================================================

print("\nCreating model...")

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
)


model = model.to(DEVICE)


# ============================================================
# 10. Print model information
# ============================================================

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Parameters: "
    f"{number_of_parameters:,}"
)

print(
    f"Device: {DEVICE}"
)


# ============================================================
# 11. Loss function
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0,
)


# ============================================================
# 12. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 13. Training function
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    for batch_index, batch in enumerate(loader):

        source_ids = batch[
            "source_ids"
        ].to(device)

        source_attention_mask = batch[
            "source_attention_mask"
        ].to(device)

        target_ids = batch[
            "target_ids"
        ].to(device)

        target_attention_mask = batch[
            "target_attention_mask"
        ].to(device)

        # ----------------------------------------------------
        # Teacher forcing
        # ----------------------------------------------------

        decoder_input = target_ids[:, :-1]

        labels = target_ids[:, 1:]

        decoder_attention_mask = (
            target_attention_mask[:, :-1]
        )

        # ----------------------------------------------------
        # Padding masks
        #
        # PyTorch Transformer expects:
        #
        # True  = ignore this token
        # False = valid token
        # ----------------------------------------------------

        source_padding_mask = (
            source_attention_mask == 0
        )

        target_padding_mask = (
            decoder_attention_mask == 0
        )

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        target_length = decoder_input.size(1)

        target_mask = create_causal_mask(
            target_length,
            device,
        )

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            source_ids=source_ids,

            target_ids=decoder_input,

            source_padding_mask=source_padding_mask,

            target_padding_mask=target_padding_mask,

            target_mask=target_mask,
        )

        # logits:
        #
        # [batch, target_length, vocab_size]

        # labels:
        #
        # [batch, target_length]

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            labels.reshape(-1),
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model parameters
        # ----------------------------------------------------

        optimizer.step()

        total_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if batch_index % 50 == 0:

            print(
                f"Batch "
                f"{batch_index:4d}/{len(loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    return total_loss / len(loader)


# ============================================================
# 14. Validation function
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            source_ids = batch[
                "source_ids"
            ].to(device)

            source_attention_mask = batch[
                "source_attention_mask"
            ].to(device)

            target_ids = batch[
                "target_ids"
            ].to(device)

            target_attention_mask = batch[
                "target_attention_mask"
            ].to(device)

            # ------------------------------------------------
            # Teacher forcing
            # ------------------------------------------------

            decoder_input = target_ids[:, :-1]

            labels = target_ids[:, 1:]

            decoder_attention_mask = (
                target_attention_mask[:, :-1]
            )

            # ------------------------------------------------
            # Padding masks
            # ------------------------------------------------

            source_padding_mask = (
                source_attention_mask == 0
            )

            target_padding_mask = (
                decoder_attention_mask == 0
            )

            # ------------------------------------------------
            # Causal mask
            # ------------------------------------------------

            target_length = decoder_input.size(1)

            target_mask = create_causal_mask(
                target_length,
                device,
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                source_ids=source_ids,

                target_ids=decoder_input,

                source_padding_mask=source_padding_mask,

                target_padding_mask=target_padding_mask,

                target_mask=target_mask,
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits.reshape(
                    -1,
                    VOCAB_SIZE,
                ),
                labels.reshape(-1),
            )

            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# 15. Training loop
# ============================================================
def main():
    print("\n")
    print("=" * 60)
    print("Starting training")
    print("=" * 60)

    best_validation_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):

        print("\n")
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print("-" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
        )

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            DEVICE,
        )

        print("\nResults:")

        print(
            f"Train loss:      "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            checkpoint_path = (
                CHECKPOINT_DIR
                / "best_model.pt"
            )

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "train_loss":
                        train_loss,

                    "validation_loss":
                        validation_loss,

                    "config": {
                        "vocab_size":
                            VOCAB_SIZE,

                        "d_model":
                            D_MODEL,

                        "nhead":
                            NHEAD,

                        "num_encoder_layers":
                            NUM_ENCODER_LAYERS,

                        "num_decoder_layers":
                            NUM_DECODER_LAYERS,

                        "dim_feedforward":
                            DIM_FEEDFORWARD,

                        "source_max_length":
                            SOURCE_MAX_LENGTH,

                        "target_max_length":
                            TARGET_MAX_LENGTH,
                            },
                },
                checkpoint_path,
            )

            print(
                f"\nSaved best model to:"
            )

            print(checkpoint_path)

    print("\n")
    print("=" * 60)
    print("Training complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
