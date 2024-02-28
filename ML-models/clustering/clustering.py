import torch
from torch import nn
from sklearn.cluster import KMeans
import os

# Load raw data tensors
processed_data_dir = 'data/processed'
bids_prices_tensor = torch.load(os.path.join(processed_data_dir, 'bids_prices.pt'))
asks_volumes_tensor = torch.load(os.path.join(processed_data_dir, 'asks_volumes.pt'))

# Concatenate tensors into one data matrix
data = torch.cat((bids_prices_tensor, asks_volumes_tensor), dim=1)

# Autoencoder model
class AE(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, emb_size)
        )

    def forward(self, x):
        return self.encoder(x)


model = AE(16)

# Extract embeddings
with torch.no_grad():
    embeddings = model(data)

# Apply KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(embeddings.numpy())
labels = kmeans.predict(embeddings.numpy())

print(labels)