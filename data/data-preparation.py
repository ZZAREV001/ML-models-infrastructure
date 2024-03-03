import torch
import numpy as np
import os
import csv


def load_and_pad_data(file_path, target_size):
    with open(file_path) as f:
        data = [float(row[0]) for row in csv.reader(f)]
    if len(data) < target_size:
        # Pad with zeros (or another suitable value)
        data.extend([0.0] * (target_size - len(data)))
    elif len(data) > target_size:
        # Optionally trim data here if necessary
        data = data[:target_size]
    return torch.tensor(data, dtype=torch.float32).unsqueeze(1)

# Determine the maximum size
sizes = []
for file_name in ['futures_asks.csv', 'futures_bids.csv']:
    with open(f'data/raw/{file_name}') as f:
        sizes.append(len(list(csv.reader(f))))
max_size = max(sizes)

# Process and pad data
asks_prices_tensor = load_and_pad_data('data/raw/futures_asks.csv', max_size)
bids_prices_tensor = load_and_pad_data('data/raw/futures_bids.csv', max_size)

# Get the processed data directory path
processed_dir = 'data/processed'
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Save tensors
torch.save(bids_prices_tensor, os.path.join(processed_dir, 'bids_prices.pt'))
torch.save(asks_prices_tensor, os.path.join(processed_dir, 'asks_volumes.pt'))
