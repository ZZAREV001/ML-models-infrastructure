import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Load your tensors
features_tensor = torch.load('data/processed/asks_volumes.pt')
labels_tensor = torch.load('data/processed/bids_prices.pt')

# Convert labels to binary
threshold = 0  # Define a threshold for what you consider an increase
binary_labels_tensor = (labels_tensor > threshold).float()

# Split your dataset into train and test sets using the binary labels
X_train, X_test, y_train, y_test = train_test_split(features_tensor, binary_labels_tensor, test_size=0.2, random_state=42)

# Define batch_size before creating the DataLoaders
batch_size = 64  # Adjust batch size to your needs

# Create TensorDatasets for both training and test sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders for both datasets
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create a dataset from tensors
dataset = TensorDataset(features_tensor, labels_tensor)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # Define the architecture
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # The sigmoid function will squash the output between 0 and 1
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# Assuming `X_train` is your feature matrix and `y_train` is your target tensor
input_dim = X_train.shape[1]

# Initialize the model
model = LogisticRegressionModel(input_dim)

# Define the loss function (Binary Cross Entropy)
criterion = nn.BCELoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Set the model to training mode
model.train()

# Number of epochs
epochs = 100

for epoch in range(epochs):
    for inputs, labels in train_loader:
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss. Ensure labels are the binary labels.
        loss = criterion(y_pred, labels.view(-1, 1))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Optionally print the loss
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Set the model to evaluation mode
model.eval()

# Predictions
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:  # Assuming you have a DataLoader `test_loader`
        outputs = model(inputs)
        predicted = (outputs.data > 0.5)  # Use a threshold of 0.5
        total += labels.size(0)
        correct += (predicted.view(-1).long() == labels).sum()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
