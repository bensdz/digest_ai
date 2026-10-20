import math
import os
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

SOURCE_MAX_LENGTH = 512
TARGET_MAX_LENGTH = 128

D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Batch size is per GPU/process, not the global batch size.
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

NUM_WORKERS = 0
PAD_ID = 0
VOCAB_SIZE = 16000


# ============================================================
# 2. Dataset
# ============================================================

class DigestDataset(Dataset):
    def __init__(self, file_path):
        data = torch.load(file_path, map_location="cpu")

        self.source_ids = data["source_ids"]
        self.source_attention_mask = data["source_attention_mask"]
        self.target_ids = data["target_ids"]
        self.target_attention_mask = data["target_attention_mask"]

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        return {
            "source_ids": self.source_ids[index],
            "source_attention_mask": self.source_attention_mask[index],
            "target_ids": self.target_ids[index],
            "target_attention_mask": self.target_attention_mask[index],
        }


class DistributedEvalSampler(Sampler):
    """Partitions evaluation indices without padding or duplicate examples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.world_size - 1) // self.world_size


# ============================================================
# 3. Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [max_length, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch, sequence_length, d_model]
        sequence_length = x.size(1)
        return x + self.pe[:sequence_length].unsqueeze(0)


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
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.source_position = PositionalEncoding(d_model, max_source_length)
        self.target_position = PositionalEncoding(d_model, max_target_length)

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

        self.output = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        source_ids,
        target_ids,
        source_padding_mask,
        target_padding_mask,
        target_mask,
    ):
        source = self.embedding(source_ids) * math.sqrt(self.d_model)
        source = self.source_position(source)

        memory = self.encoder(
            source,
            src_key_padding_mask=source_padding_mask,
        )

        target = self.embedding(target_ids) * math.sqrt(self.d_model)
        target = self.target_position(target)

        decoded = self.decoder(
            target,
            memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )

        return self.output(decoded)


# ============================================================
# 5. Causal mask
# ============================================================

def create_causal_mask(sequence_length, device):
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, device=device),
        diagonal=1,
    )
    return mask.masked_fill(mask == 1, float("-inf"))


# ============================================================
# 6. Distributed setup
# ============================================================

def setup_distributed():
    """
    Supports:
      - Slurm: srun launches one Python process per GPU.
      - torchrun: RANK, LOCAL_RANK and WORLD_SIZE are provided.

    The Slurm example below uses --gpus-per-task=1, so each process
    normally sees its assigned GPU as cuda:0.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        required = ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(
                "Distributed environment variables are missing: "
                + ", ".join(missing)
                + ". Launch this script with srun or torchrun."
            )

        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. DDP script expects NVIDIA GPUs and "
            "a CUDA-enabled PyTorch installation."
        )

    # When Slurm gives each task exactly one GPU, it is exposed as cuda:0.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_gpu_count = len(
        [item for item in visible_devices.split(",") if item.strip()]
    )
    device_index = 0 if visible_gpu_count == 1 else local_rank

    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{device_index}, but this process sees "
            f"{torch.cuda.device_count()} CUDA device(s). Check Slurm GPU binding."
        )

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    return rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# 7. Training and validation
# ============================================================

def move_batch_to_device(batch, device):
    source_ids = batch["source_ids"].to(device, non_blocking=True)
    source_attention_mask = batch["source_attention_mask"].to(
        device, non_blocking=True
    )
    target_ids = batch["target_ids"].to(device, non_blocking=True)
    target_attention_mask = batch["target_attention_mask"].to(
        device, non_blocking=True
    )

    return (
        source_ids,
        source_attention_mask,
        target_ids,
        target_attention_mask,
    )


def train_one_epoch(model, loader, optimizer, device, rank):
    model.train()

    # Track token-weighted loss so metrics can be correctly combined
    # across processes even if their last batches have different sizes.
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.float64, device=device)

    for batch_index, batch in enumerate(loader):
        (
            source_ids,
            source_attention_mask,
            target_ids,
            target_attention_mask,
        ) = move_batch_to_device(batch, device)

        # Teacher forcing: predict each target token from earlier tokens.
        decoder_input = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        decoder_attention_mask = target_attention_mask[:, :-1]

        source_padding_mask = source_attention_mask == 0
        target_padding_mask = decoder_attention_mask == 0
        target_mask = create_causal_mask(decoder_input.size(1), device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            source_ids=source_ids,
            target_ids=decoder_input,
            source_padding_mask=source_padding_mask,
            target_padding_mask=target_padding_mask,
            target_mask=target_mask,
        )

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
            ignore_index=PAD_ID,
        )

        loss.backward()
        optimizer.step()

        valid_tokens = (labels != PAD_ID).sum()
        total_loss_sum += loss.detach().to(torch.float64) * valid_tokens
        total_tokens += valid_tokens

        if rank == 0 and batch_index % 50 == 0:
            print(
                f"Batch {batch_index:4d}/{len(loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True,
            )

    # Every process contributes its total loss and token count.
    stats = torch.stack([total_loss_sum, total_tokens])
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    return (stats[0] / stats[1].clamp_min(1)).item()


@torch.no_grad()
def validate(model, loader, device):
    """
    Run validation using the unwrapped model (model.module) and a sampler
    that does not duplicate examples. This avoids DDP forward collectives
    when ranks have different numbers of validation batches.
    """
    model.eval()

    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.float64, device=device)

    for batch in loader:
        (
            source_ids,
            source_attention_mask,
            target_ids,
            target_attention_mask,
        ) = move_batch_to_device(batch, device)

        decoder_input = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        decoder_attention_mask = target_attention_mask[:, :-1]

        source_padding_mask = source_attention_mask == 0
        target_padding_mask = decoder_attention_mask == 0
        target_mask = create_causal_mask(decoder_input.size(1), device)

        logits = model(
            source_ids=source_ids,
            target_ids=decoder_input,
            source_padding_mask=source_padding_mask,
            target_padding_mask=target_padding_mask,
            target_mask=target_mask,
        )

        loss_sum = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
            ignore_index=PAD_ID,
            reduction="sum",
        )

        valid_tokens = (labels != PAD_ID).sum()
        total_loss_sum += loss_sum.to(torch.float64)
        total_tokens += valid_tokens

    stats = torch.stack([total_loss_sum, total_tokens])
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    return (stats[0] / stats[1].clamp_min(1)).item()


# ============================================================
# 8. Main
# ============================================================

def main():
    rank, local_rank, world_size, device = setup_distributed()

    try:
        if rank == 0:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            print("=" * 60)
            print("Digest DDP training")
            print(f"Processes / GPUs: {world_size}")
            print(f"Per-GPU batch size: {BATCH_SIZE}")
            print(f"Effective global batch size: {BATCH_SIZE * world_size}")
            print(f"Device on rank 0: {device}")
            print("=" * 60, flush=True)

        # Ensure rank 0 has created the checkpoint directory before
        # the other ranks continue (assuming a shared filesystem).
        dist.barrier()

        train_dataset = DigestDataset(PROCESSED_DIR / "train_tensors.pt")
        validation_dataset = DigestDataset(
            PROCESSED_DIR / "validation_tensors.pt"
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

        validation_sampler = DistributedEvalSampler(
            validation_dataset,
            rank=rank,
            world_size=world_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            sampler=validation_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

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
        ).to(device)

        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        if rank == 0:
            parameter_count = sum(
                parameter.numel() for parameter in model.module.parameters()
            )
            print(f"Parameters: {parameter_count:,}", flush=True)
            print(f"Training examples: {len(train_dataset)}", flush=True)
            print(
                f"Validation examples: {len(validation_dataset)}",
                flush=True,
            )

        best_validation_loss = float("inf")

        for epoch in range(1, NUM_EPOCHS + 1):
            # Required for a different distributed shuffle each epoch.
            train_sampler.set_epoch(epoch)

            if rank == 0:
                print(f"\nEpoch {epoch}/{NUM_EPOCHS}", flush=True)

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                rank=rank,
            )

            # Validation sampler does not pad/duplicate examples. Use the
            # underlying model so uneven validation batch counts cannot
            # cause DDP forward synchronization problems.
            validation_loss = validate(
                model=model.module,
                loader=validation_loader,
                device=device,
            )

            if rank == 0:
                print(
                    f"Train loss: {train_loss:.4f} | "
                    f"Validation loss: {validation_loss:.4f}",
                    flush=True,
                )

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "validation_loss": validation_loss,
                            "config": {
                                "vocab_size": VOCAB_SIZE,
                                "d_model": D_MODEL,
                                "nhead": NHEAD,
                                "num_encoder_layers": NUM_ENCODER_LAYERS,
                                "num_decoder_layers": NUM_DECODER_LAYERS,
                                "dim_feedforward": DIM_FEEDFORWARD,
                                "dropout": DROPOUT,
                                "source_max_length": SOURCE_MAX_LENGTH,
                                "target_max_length": TARGET_MAX_LENGTH,
                            },
                        },
                        checkpoint_path,
                    )
                    print(
                        f"Saved best checkpoint: {checkpoint_path}", flush=True)

            # Keep all ranks in step before starting the next epoch.
            dist.barrier()

        if rank == 0:
            print("\nTraining complete.", flush=True)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
