import torch
import torch.nn as nn
import os

# Load preprocessed data
processed_data_dir = 'data/processed'
bids_prices_tensor = torch.load(os.path.join(processed_data_dir, 'bids_prices.pt'))
asks_volumes_tensor = torch.load(os.path.join(processed_data_dir, 'asks_volumes.pt'))


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        return self.linear(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, emb_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Model
class Autoencoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.encoder = Encoder(28 * 28, emb_dim)
        self.decoder = Decoder(emb_dim, 28 * 28)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder(64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, X)
    loss.backward()
    optimizer.step()

    print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, loss.item()))
