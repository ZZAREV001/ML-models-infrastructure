import torch
from torch import nn
from sklearn.cluster import KMeans
from model import Autoencoder, Encoder, Decoder
import os

# Load preprocessed data
processed_data_dir = 'data/processed'
bids_prices_tensor = torch.load(os.path.join(processed_data_dir, 'bids_prices.pt'))
asks_volumes_tensor = torch.load(os.path.join(processed_data_dir, 'asks_volumes.pt'))

# Autoencoder for embeddings
class AE(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, emb_size)
        )

    def forward(self, x):
        return self.encoder(x)


model = AE(16)

# Extract embeddings
with torch.no_grad():
    embeddings = model(X)

# Apply KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(embeddings.numpy())
labels = kmeans.predict(embeddings.numpy())

print(labels)
