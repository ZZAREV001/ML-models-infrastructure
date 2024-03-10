import torch
import torch.nn as nn
import numpy as np
from global_analysis.ML_models.classification.data_preprocessor import DataPreprocessor


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.view(-1, 1)  # Reshape the output to match the target shape
        return out


# Set the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, Type: {type(device)}")

# Define model parameters
hidden_size = 64  # Example hidden layer size
num_layers = 2  # Example number of layers
num_epochs = 100  # Example number of epochs
batch_size = 32  # Example batch size

# Specify the file paths for each CSV file
file_paths = {
    'open_interest': '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/open_interest.csv',
    'bid_data': '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/futures_bids.csv',
    'ask_data': '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/futures_asks.csv',
    'recent_trades': '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/recentTrades.csv'
}

# Create an instance of the DataPreprocessor with the updated paths
preprocessor = DataPreprocessor(file_paths)

# Load and preprocess the data
train_data, test_data, train_targets, test_targets = preprocessor.load_and_preprocess_data()

# Convert data and targets to PyTorch tensors and move them to the device
train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)

# Reshape the target tensors
train_targets = train_targets.view(-1, 1)
test_targets = test_targets.view(-1, 1)

print(f"Train Data Shape: {train_data.shape}")
print(f"Train Targets Shape: {train_targets.shape}")
print(f"Test Data Shape: {test_data.shape}")
print(f"Test Targets Shape: {test_targets.shape}")

# The input_size should be updated based on the new structure of your data
input_size = train_data.shape[-1]  # This will automatically match the new feature count

# Assuming the target selection is updated in the DataPreprocessor,
# output_size remains 1 if predicting a single value
output_size = 1

# Create an instance of the LSTM model with the updated input size
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)


# Training and evaluation function
def train_and_evaluate(model, train_data, train_targets, test_data, test_targets, num_epochs, batch_size):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch_x = train_data[i:i + batch_size]
            batch_y = train_targets[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_data):.4f}')

    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        mse = nn.MSELoss()(outputs, test_targets)
        print(f"Mean Squared Error (MSE): {mse:.4f}")


# Train and evaluate the model
train_and_evaluate(model, train_data, test_data, train_targets, test_targets, num_epochs, batch_size)