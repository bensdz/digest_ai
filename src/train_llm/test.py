from generate import (
    model,
    tokenizer,
    generate_summary,
    DEVICE,
    PAD_ID,
    BOS_ID,
    EOS_ID,
    PROJECT_ROOT,
    create_causal_mask,
)
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_tensors.pt"
)

BATCH_SIZE = 4


# --------------------------------------------------
# 2. Load the unseen test dataset
# --------------------------------------------------

test_data = torch.load(
    TEST_DATA_PATH,
    map_location="cpu",
    weights_only=True,
)

test_dataset = TensorDataset(
    test_data["source_ids"],
    test_data["source_attention_mask"],
    test_data["target_ids"],
    test_data["target_attention_mask"],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Test examples:", len(test_dataset))
print("Device:", DEVICE)


# --------------------------------------------------
# 3. Calculate test loss
# --------------------------------------------------

def evaluate_test_loss():
    model.eval()

    # Sum token losses so we can calculate a
    # token-weighted average across all batches.
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        reduction="sum",
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for (
            source_ids,
            source_attention_mask,
            target_ids,
            target_attention_mask,
        ) in test_loader:

            source_ids = source_ids.to(DEVICE)
            source_attention_mask = source_attention_mask.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            target_attention_mask = target_attention_mask.to(DEVICE)

            # Teacher forcing: shift target by one position.
            decoder_input = target_ids[:, :-1]
            labels = target_ids[:, 1:]
            decoder_attention_mask = target_attention_mask[:, :-1]

            source_padding_mask = source_attention_mask == 0
            target_padding_mask = decoder_attention_mask == 0

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

            batch_loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            total_loss += batch_loss.item()
            total_tokens += (labels != PAD_ID).sum().item()

    average_loss = total_loss / max(total_tokens, 1)

    return average_loss


# --------------------------------------------------
# 4. Decode a reference summary
# --------------------------------------------------

def decode_reference(token_ids):
    # Remove padding and sequence boundary tokens.
    cleaned_ids = [
        int(token_id)
        for token_id in token_ids
        if int(token_id) not in (PAD_ID, BOS_ID, EOS_ID)
    ]

    return tokenizer.decode(cleaned_ids)


# --------------------------------------------------
# 5. Inspect generated summaries
# --------------------------------------------------

def inspect_examples(number_of_examples=5):
    print("\n" + "=" * 70)
    print("GENERATED SUMMARY COMPARISON")
    print("=" * 70)

    for index in range(min(number_of_examples, len(test_dataset))):
        example = test_dataset[index]

        source_ids = example[0]
        target_ids = example[2]

        # Recover the source text from its token IDs.
        source_token_ids = [
            int(token_id)
            for token_id in source_ids.tolist()
            if int(token_id) not in (PAD_ID, BOS_ID, EOS_ID)
        ]

        source_text = tokenizer.decode(source_token_ids)
        reference_summary = decode_reference(target_ids.tolist())

        # generate_summary performs inference only.
        predicted_summary = generate_summary(source_text)

        print(f"\nEXAMPLE {index + 1}")
        print("-" * 70)
        print("SOURCE:")
        print(source_text)

        print("\nREFERENCE SUMMARY:")
        print(reference_summary)

        print("\nGENERATED SUMMARY:")
        print(predicted_summary)


# --------------------------------------------------
# 6. Run evaluation
# --------------------------------------------------

if __name__ == "__main__":
    test_loss = evaluate_test_loss()

    print(f"\nTest loss: {test_loss:.4f}")

    inspect_examples(number_of_examples=5)
