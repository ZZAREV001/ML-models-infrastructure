import torch
from torch import nn
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Assuming `data` is your dataset and `labels` are the cluster labels from KMeans
data_numpy = data.numpy()  # Convert PyTorch tensor to numpy array if necessary

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_numpy)

# Plotting
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w')
plt.title('Cluster Visualization with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=scatter.legend_elements()[0], title="Clusters")
plt.show()