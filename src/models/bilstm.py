import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ISLBiLSTM(nn.Module):
    def __init__(
        self,
        input_size=225,
        hidden_size=128,
        num_layers=2,
        num_classes=50,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x, lengths):
        """
        x:
            Padded input tensor
            Shape: [batch, max_sequence_length, 225]

        lengths:
            Actual sequence lengths
            Shape: [batch]
        """

        # Convert padded sequences into a packed representation.
        packed_x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Run the packed sequences through the BiLSTM.
        _, (hidden, _) = self.lstm(packed_x)

        # Last forward hidden state
        forward_hidden = hidden[-2]

        # Last backward hidden state
        backward_hidden = hidden[-1]

        # Combine both directions
        final_hidden = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        # Classification
        logits = self.classifier(final_hidden)

        return logits