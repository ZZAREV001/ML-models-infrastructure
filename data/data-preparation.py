import torch
import pandas as pd
import numpy as np
import os
import csv

with open('data/raw/futures_asks.csv') as f:
    asks_prices = [float(row[0]) for row in csv.reader(f)]

asks_prices = np.array(asks_prices)
asks_prices_tensor = torch.from_numpy(asks_prices)

with open('data/raw/futures_bids.csv') as f:
    bids_prices = [float(row[0]) for row in csv.reader(f)]

bids_prices = np.array(bids_prices)
bids_prices_tensor = torch.from_numpy(bids_prices)

# Get the processed data directory path
processed_dir = 'data/processed'
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Save tensors
torch.save(bids_prices_tensor, os.path.join(processed_dir, 'bids_prices.pt'))
torch.save(asks_prices_tensor, os.path.join(processed_dir, 'asks_volumes.pt'))



