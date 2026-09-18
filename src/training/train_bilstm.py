import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.training.include_dataset import create_dataloaders
from src.models.bilstm import ISLBiLSTM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "include50_bilstm_best.pth"
)

OUTPUT_DIR = Path("data/evaluation/include50_bilstm")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Dataset diagnostics / class balancing
# --------------------------------------------------

def get_class_counts(loader, num_classes):
    counts = Counter()

    for _, _, labels in loader:
        counts.update(labels.tolist())

    return np.array(
        [counts.get(class_id, 0) for class_id in range(num_classes)],
        dtype=np.int64
    )


def print_class_distribution(
    loader,
    split_name,
    id_to_label,
    output_dir=OUTPUT_DIR
):
    num_classes = len(id_to_label)
    counts = get_class_counts(loader, num_classes)

    print(f"\n{split_name} class distribution")
    print("-" * 70)

    for class_id, count in enumerate(counts):
        label = id_to_label[class_id]
        print(f"{class_id:2d} | {str(label):30s} | {count:3d}")

    zero_classes = np.where(counts == 0)[0].tolist()

    if zero_classes:
        print(
            f"\nClasses with ZERO {split_name.lower()} support: "
            f"{zero_classes}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "class_id": np.arange(num_classes),
        "label": [id_to_label[i] for i in range(num_classes)],
        "count": counts,
    }).to_csv(
        output_dir / f"{split_name.lower()}_class_distribution.csv",
        index=False,
    )

    return counts


def get_class_weights(loader, num_classes):
    """Gentle inverse-frequency weighting using 1/sqrt(count)."""
    counts = get_class_counts(loader, num_classes).astype(np.float32)

    # Do not let an absent training class produce an infinite weight.
    safe_counts = np.maximum(counts, 1.0)

    weights = 1.0 / np.sqrt(safe_counts)

    # Normalize so average class weight is approximately 1.
    weights = weights / weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
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

        optimizer.zero_grad()

        logits = model(sequences, lengths)
        loss = criterion(logits, labels)

        loss.backward()
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
# Detailed test evaluation
# --------------------------------------------------

def evaluate_detailed(
    model,
    loader,
    device,
    id_to_label,
    num_classes=50,
):
    model.eval()

    all_labels = []
    all_preds = []
    all_logits = []

    total_loss = 0.0
    total_samples = 0

    # Keep test loss unweighted so it remains a standard cross-entropy metric.
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(sequences, lengths)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_logits.append(logits.cpu())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    logits = torch.cat(all_logits, dim=0)

    # --------------------------------------------------
    # Top-1 / Top-5 accuracy
    # --------------------------------------------------
    top1_accuracy = accuracy_score(y_true, y_pred)

    top5_predictions = logits.topk(5, dim=1).indices.numpy()

    top5_accuracy = np.mean([
        true_label in predicted_labels
        for true_label, predicted_labels
        in zip(y_true, top5_predictions)
    ])

    # --------------------------------------------------
    # Only classes that actually occur in the test set
    # --------------------------------------------------
    present_classes = np.unique(y_true)

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=present_classes,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=present_classes,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=present_classes,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    test_loss = total_loss / total_samples

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    # --------------------------------------------------
    # Prediction-count diagnostics
    # --------------------------------------------------
    true_counts = np.bincount(
        y_true,
        minlength=num_classes
    )

    pred_counts = np.bincount(
        y_pred,
        minlength=num_classes
    )

    prediction_counts = pd.DataFrame({
        "class_id": np.arange(num_classes),
        "label": [id_to_label[i] for i in range(num_classes)],
        "true_count": true_counts,
        "predicted_count": pred_counts,
    })

    # --------------------------------------------------
    # Per-class report, excluding classes with zero support
    # --------------------------------------------------
    present_labels = [id_to_label[int(i)] for i in present_classes]

    report = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=present_labels,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=present_labels,
        output_dict=True,
        zero_division=0,
    )

    # Lowest-F1 supported classes
    worst_classes = []

    for class_id in present_classes:
        label = id_to_label[int(class_id)]
        metrics = report_dict[label]

        worst_classes.append({
            "class_id": int(class_id),
            "label": label,
            "f1": metrics["f1-score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "support": int(metrics["support"]),
        })

    worst_classes = sorted(
        worst_classes,
        key=lambda item: item["f1"]
    )

    return {
        "loss": test_loss,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "present_classes": present_classes,
        "prediction_counts": prediction_counts,
        "report": report,
        "worst_classes": worst_classes,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Seed:", SEED)

    # --------------------------------------------------
    # Create datasets and DataLoaders
    # --------------------------------------------------
    (
        train_loader,
        val_loader,
        test_loader,
        label_to_id,
        id_to_label,
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    num_classes = len(label_to_id)

    # --------------------------------------------------
    # Dataset diagnostics
    # --------------------------------------------------
    train_counts = print_class_distribution(
        train_loader,
        "TRAIN",
        id_to_label,
    )

    val_counts = print_class_distribution(
        val_loader,
        "VAL",
        id_to_label,
    )

    test_counts = print_class_distribution(
        test_loader,
        "TEST",
        id_to_label,
    )

    missing_test_classes = np.where(test_counts == 0)[0].tolist()

    if missing_test_classes:
        print(
            "\nWARNING: The following classes have zero test support: "
            f"{missing_test_classes}"
        )
        print(
            "These classes cannot have meaningful test recall/F1 and "
            "are excluded from supported-class macro metrics."
        )

    # --------------------------------------------------
    # Class weights for training only
    # --------------------------------------------------
    class_weights = get_class_weights(
        train_loader,
        num_classes,
    )

    print("\nClass weights:")
    print(class_weights.detach().cpu().numpy())

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------
    model = ISLBiLSTM(
        input_size=225,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    )

    model = model.to(DEVICE)

    print("\nModel:")
    print(model)

    # --------------------------------------------------
    # Loss functions
    # --------------------------------------------------
    # Weighted loss helps rare classes during training.
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Unweighted validation loss keeps model selection/evaluation interpretable.
    eval_criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            eval_criterion,
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

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
                    "num_classes": num_classes,
                    "dropout": 0.3,
                    "seed": SEED,
                    "class_weights": class_weights.detach().cpu(),
                },
                BEST_MODEL_PATH,
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
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------
    # Final detailed test evaluation
    # --------------------------------------------------
    results = evaluate_detailed(
        model,
        test_loader,
        DEVICE,
        id_to_label,
        num_classes=num_classes,
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test Loss:       {results['loss']:.4f}")
    print(
        f"Top-1 Accuracy:  "
        f"{results['top1_accuracy']:.4f} "
        f"({results['top1_accuracy'] * 100:.2f}%)"
    )
    print(
        f"Top-5 Accuracy:  "
        f"{results['top5_accuracy']:.4f} "
        f"({results['top5_accuracy'] * 100:.2f}%)"
    )
    print(
        f"Macro Precision: "
        f"{results['macro_precision']:.4f} "
        f"({results['macro_precision'] * 100:.2f}%)"
    )
    print(
        f"Macro Recall:    "
        f"{results['macro_recall']:.4f} "
        f"({results['macro_recall'] * 100:.2f}%)"
    )
    print(
        f"Macro F1:        "
        f"{results['macro_f1']:.4f} "
        f"({results['macro_f1'] * 100:.2f}%)"
    )
    print(
        f"Weighted F1:     "
        f"{results['weighted_f1']:.4f} "
        f"({results['weighted_f1'] * 100:.2f}%)"
    )
    print(
        f"Supported test classes: "
        f"{len(results['present_classes'])}/{num_classes}"
    )

    if missing_test_classes:
        print(
            f"Classes with zero test support: "
            f"{missing_test_classes}"
        )

    # --------------------------------------------------
    # Save evaluation outputs
    # --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cm = results["confusion_matrix"]

    class_names = [id_to_label[i] for i in range(num_classes)]

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    plt.figure(figsize=(14, 12))
    plt.imshow(
        cm,
        interpolation="nearest",
        aspect="auto",
    )
    plt.title("INCLUDE-50 BiLSTM Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar()

    plt.xticks(
        np.arange(num_classes),
        class_names,
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        np.arange(num_classes),
        class_names,
        fontsize=7,
    )

    # Annotate only non-zero cells.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    results["prediction_counts"].to_csv(
        OUTPUT_DIR / "prediction_counts.csv",
        index=False,
    )

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(
        results["report"],
        encoding="utf-8",
    )

    worst_df = pd.DataFrame(results["worst_classes"])
    worst_df.to_csv(
        OUTPUT_DIR / "worst_classes_by_f1.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "seed": SEED,
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": results["loss"],
        "top1_accuracy": results["top1_accuracy"],
        "top5_accuracy": results["top5_accuracy"],
        "macro_precision_supported": results["macro_precision"],
        "macro_recall_supported": results["macro_recall"],
        "macro_f1_supported": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "supported_test_classes": len(results["present_classes"]),
        "total_classes": num_classes,
    }])

    summary.to_csv(
        OUTPUT_DIR / "evaluation_summary.csv",
        index=False,
    )

    print("\nLowest-F1 supported classes")
    print("---------------------------")
    for item in results["worst_classes"][:10]:
        print(
            f"Class {item['class_id']:2d} | "
            f"{str(item['label']):25s} | "
            f"F1: {item['f1']:.3f} | "
            f"Support: {item['support']} | "
            f"Predicted: {int(results['prediction_counts'].loc[results['prediction_counts']['class_id'] == item['class_id'], 'predicted_count'].iloc[0])}"
        )

    print(
        f"\nConfusion matrix saved to: "
        f"{OUTPUT_DIR / 'confusion_matrix.png'}"
    )
    print(
        f"Evaluation summary saved to: "
        f"{OUTPUT_DIR / 'evaluation_summary.csv'}"
    )
    print(
        f"Best model saved to:\n{BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
