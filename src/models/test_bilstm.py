import torch
from models.bilstm import ISLBiLSTM


model = ISLBiLSTM()

# Simulate a DataLoader batch
batch_size = 4
max_sequence_length = 71
feature_size = 225

x = torch.randn(
    batch_size,
    max_sequence_length,
    feature_size
)

lengths = torch.tensor([
    61,
    71,
    44,
    71
])

# Forward pass
output = model(x, lengths)

print("Model:")
print(model)

print("\nInput shape:")
print(x.shape)

print("\nSequence lengths:")
print(lengths)

print("\nOutput shape:")
print(output.shape)