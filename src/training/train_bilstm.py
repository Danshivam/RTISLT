import os
import torch
import torch.nn as nn
from torch.optim import Adam

from include_dataset import create_dataloaders
from models.bilstm import ISLBiLSTM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "include50_bilstm_best.pth"
)


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, lengths, labels in loader:

        sequences = sequences.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(sequences, lengths)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy


# --------------------------------------------------
# Validation function
# --------------------------------------------------

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for sequences, lengths, labels in loader:

            sequences = sequences.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(sequences, lengths)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(loader)
    accuracy = correct / total

    return average_loss, accuracy


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Device:", DEVICE)

    # Create datasets and DataLoaders
    train_loader, val_loader, test_loader, label_to_id, id_to_label = \
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=0
        )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Create model
    model = ISLBiLSTM(
        input_size=225,
        hidden_size=128,
        num_layers=2,
        num_classes=len(label_to_id),
        dropout=0.3
    )

    model = model.to(DEVICE)

    print("\nModel:")
    print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "input_size": 225,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "num_classes": len(label_to_id)
                },
                BEST_MODEL_PATH
            )

            print(
                f"  -> Best model saved "
                f"(val_acc={val_accuracy:.4f})"
            )

    # --------------------------------------------------
    # Load best model
    # --------------------------------------------------

    print("\nLoading best model...")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(
        f"\nBest model saved to:\n{BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()