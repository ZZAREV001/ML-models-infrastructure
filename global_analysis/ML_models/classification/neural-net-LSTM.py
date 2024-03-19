import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define the DataPreprocessor class
class DataPreprocessor:
    def __init__(self, merged_data_file):
        self.merged_data_file = merged_data_file

    def load_and_preprocess_data(self):
        # Load the merged data CSV file
        merged_data = pd.read_csv(self.merged_data_file)

        print("Merged Dataset:")
        print(merged_data.head())
        print("Merged Dataset Shape:", merged_data.shape)

        # Select relevant features from the merged dataset, excluding 'Value'
        features = ['open', 'high', 'low', 'close', 'volume', 'OpenInterest']

        # Preprocess the merged dataset
        data = merged_data[features]
        data = data.dropna()  # Remove rows with missing values

        print("Preprocessed Data:")
        print(data.head())
        print("Preprocessed Data Shape:", data.shape)

        # Scale the input features
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        print("Scaled Data Shape:", data_scaled.shape)

        # Adjust the target for your model since 'Value' is no longer available
        # Decide on a new target based on available columns or use an external target if applicable

        # Create input sequences and targets with the adjusted target
        sequence_length = 2  # Keep or adjust the sequence length as needed
        X = []
        y = []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i + sequence_length])
            # Example for 'y': Use 'OpenInterest' as the new target
            y.append(data_scaled[i + sequence_length, features.index('OpenInterest')])
        X = np.array(X)
        y = np.array(y)

        print("Input Sequences Shape:", X.shape)
        print("Targets Shape:", y.shape)

        # Check if the input sequences and targets are empty
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input sequences or targets are empty. Please adjust the sequence length or check the input data.")

        # Split the data into training and testing sets
        train_data, test_data, train_targets, test_targets = train_test_split(X, y, test_size=0.2, random_state=42)

        return train_data, test_data, train_targets, test_targets

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
        return out

# Set the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, Type: {type(device)}")

# Define model parameters
hidden_size = 64  # Example hidden layer size
num_layers = 2  # Example number of layers
num_epochs = 100  # Example number of epochs
batch_size = 32  # Example batch size

# Specify the file path for the merged data CSV file
merged_data_file = 'global_analysis/processed/merged_data.csv'

# Create an instance of the DataPreprocessor with the merged data file
preprocessor = DataPreprocessor(merged_data_file)

# Load and preprocess the data
train_data, test_data, train_targets, test_targets = preprocessor.load_and_preprocess_data()

# Convert data and targets to PyTorch tensors and move them to the device
train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)

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
        num_batches = 0
        for i in range(0, len(train_data), batch_size):
            batch_x = train_data[i:i + batch_size]
            batch_y = train_targets[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / num_batches:.4f}')

    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        mse = nn.MSELoss()(outputs, test_targets)
        print(f"Mean Squared Error (MSE): {mse:.4f}")

# Train and evaluate the model
train_and_evaluate(model, train_data, train_targets, test_data, test_targets, num_epochs, batch_size)