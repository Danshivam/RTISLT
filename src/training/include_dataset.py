import os

import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = (
    "data/include/metadata/include50_manifest.csv"
)

LANDMARK_ROOT = (
    "data/landmarks/include50"
)


# ============================================================
# LABEL ENCODING
# ============================================================

def create_label_mapping(manifest):
    """
    Create a mapping:

        sign name -> integer class

    Example:

        "1. loud"  -> 0
        "2. quiet" -> 1
        ...
    """

    labels = sorted(
        manifest["label"]
        .unique()
    )

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index
        in label_to_id.items()
    }

    return label_to_id, id_to_label


# ============================================================
# BUILD LANDMARK FILE PATH
# ============================================================

def build_landmark_path(row):
    """
    Recreate the exact .npy filename produced
    by preprocess_include.py.
    """

    relative_video = str(
        row["video_path"]
    )

    split = str(
        row["split"]
    )

    label = str(
        row["label"]
    )

    parent = str(
        row["parent_label"]
    )

    # --------------------------------------------
    # Make safe filename components
    # --------------------------------------------

    safe_parent = (
        parent
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    safe_label = (
        label
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "")
    )

    video_name = os.path.splitext(
        os.path.basename(
            relative_video
        )
    )[0]

    filename = (
        f"{safe_parent}_"
        f"{safe_label}_"
        f"{video_name}.npy"
    )

    return os.path.join(
        LANDMARK_ROOT,
        split,
        filename
    )


# ============================================================
# DATASET
# ============================================================

class INCLUDE50Dataset(Dataset):

    def __init__(
        self,
        manifest,
        split,
        label_to_id
    ):
        """
        Dataset for one INCLUDE-50 split.

        split:
            train
            val
            test
        """

        self.data = manifest[
            manifest["split"] == split
        ].reset_index(
            drop=True
        )

        self.label_to_id = label_to_id

        print()
        print(
            f"{split.upper()} samples:",
            len(self.data)
        )


    def __len__(self):

        return len(
            self.data
        )


    def __getitem__(self, index):

        row = self.data.iloc[index]

        # --------------------------------------------
        # Landmark file
        # --------------------------------------------

        landmark_path = build_landmark_path(
            row
        )

        # --------------------------------------------
        # Safety check
        # --------------------------------------------

        if not os.path.isfile(
            landmark_path
        ):

            raise FileNotFoundError(
                f"Landmark file not found:\n"
                f"{landmark_path}"
            )

        # --------------------------------------------
        # Load sequence
        # --------------------------------------------

        sequence = np.load(
            landmark_path
        )

        # --------------------------------------------
        # Verify shape
        # --------------------------------------------

        if sequence.ndim != 2:

            raise ValueError(
                f"Expected 2D sequence, "
                f"got {sequence.shape}"
            )

        if sequence.shape[1] != 225:

            raise ValueError(
                f"Expected 225 features, "
                f"got {sequence.shape}"
            )

        # --------------------------------------------
        # Convert NumPy → PyTorch
        # --------------------------------------------

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        # --------------------------------------------
        # Label
        # --------------------------------------------

        label_name = str(
            row["label"]
        )

        label_id = self.label_to_id[
            label_name
        ]

        label = torch.tensor(
            label_id,
            dtype=torch.long
        )

        return (
            sequence,
            label
        )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_batch(batch):
    """
    Convert variable-length sequences into
    one padded batch.

    Example:

        (56,225)
        (81,225)
        (65,225)

    becomes:

        (3,81,225)

    Returns:

        padded_sequences
        lengths
        labels
    """

    sequences, labels = zip(
        *batch
    )

    # --------------------------------------------
    # Original sequence lengths
    # --------------------------------------------

    lengths = torch.tensor(
        [
            sequence.shape[0]
            for sequence in sequences
        ],
        dtype=torch.long
    )

    # --------------------------------------------
    # Pad sequences
    # --------------------------------------------

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    # --------------------------------------------
    # Stack labels
    # --------------------------------------------

    labels = torch.stack(
        labels
    )

    return (
        padded_sequences,
        lengths,
        labels
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=16,
    num_workers=0
):

    # --------------------------------------------
    # Load manifest
    # --------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # --------------------------------------------
    # Create label mapping
    # --------------------------------------------

    label_to_id, id_to_label = (
        create_label_mapping(
            manifest
        )
    )

    print()
    print(
        "Number of classes:",
        len(label_to_id)
    )

    # --------------------------------------------
    # Create datasets
    # --------------------------------------------

    train_dataset = INCLUDE50Dataset(
        manifest,
        "train",
        label_to_id
    )

    val_dataset = INCLUDE50Dataset(
        manifest,
        "val",
        label_to_id
    )

    test_dataset = INCLUDE50Dataset(
        manifest,
        "test",
        label_to_id
    )

    # --------------------------------------------
    # DataLoaders
    # --------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        label_to_id,
        id_to_label
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader, \
        label_to_id, id_to_label = create_dataloaders(
            batch_size=4
        )

    # --------------------------------------------
    # Get one training batch
    # --------------------------------------------

    sequences, lengths, labels = next(
        iter(train_loader)
    )

    print()
    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print(
        "Padded sequence shape:",
        sequences.shape
    )

    print(
        "Sequence lengths:",
        lengths.tolist()
    )

    print(
        "Labels:",
        labels.tolist()
    )

    print(
        "Number of classes:",
        len(label_to_id)
    )

    print(
        "First class:",
        id_to_label[0],
        

    )

    print("=" * 60)