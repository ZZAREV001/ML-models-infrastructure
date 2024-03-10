using Flux
using Flux: @functor, chunk, throttle, logitcrossentropy
using CUDA
using CSV
using DataFrames
using Dates
using MLDataUtils: shuffleobs, splitobs
using MLDataUtils
using Statistics

# Define the LSTM model
struct LSTMModel
    lstm::LSTM
    fc::Dense
end

@functor LSTMModel

function (m::LSTMModel)(x)
    out, _ = m.lstm(x)
    out = m.fc(out[:, end, :])
    return out
end

# Set the device (GPU if available, else CPU)
device = gpu

# Load the dataset
data = CSV.File("/Users/GoldenEagle/Downloads/BTCUSDT-trades-2024-03-08.csv") |> DataFrame

# Convert timestamps to datetime format
data[:, :time] = DateTime.(data[:, :time])

# Select relevant features
features = [:price, :qty, :quote_qty, :is_buyer_maker]

# Convert 'is_buyer_maker' to numeric representation
data[:, :is_buyer_maker] = convert.(Int, data[:, :is_buyer_maker])

# Create input sequences
sequence_length = 10  # Adjust the sequence length as needed
X = []
y = []
for i in 1:(size(data, 1) - sequence_length)
    push!(X, convert(Array, data[i:i+sequence_length-1, features]))
    push!(y, data[i+sequence_length, :price])
end
X = hcat(X...)
y = convert(Array, y)

# Scale the input features
X_scaled = (X .- mean(X, dims=2)) ./ std(X, dims=2)

# Split the data into training and testing sets
train_data, test_data, train_targets, test_targets = splitobs((X_scaled, y); at=0.8)

# Convert data to Flux arrays and move to the device
train_data = train_data |> gpu
train_targets = train_targets |> gpu
test_data = test_data |> gpu
test_targets = test_targets |> gpu

# Define the model parameters
input_size = size(train_data, 1)
hidden_size = 64
num_layers = 2
output_size = 1

# Create an instance of the LSTM model
model = LSTMModel(
    LSTM(input_size, hidden_size; numLayers=num_layers),
    Dense(hidden_size, output_size)
) |> gpu

# Define the loss function and optimizer
loss(x, y) = Flux.mse(model(x), y)
optimizer = ADAM()

# Training loop
num_epochs = 100
batch_size = 32

for epoch in 1:num_epochs
    for i in 1:batch_size:size(train_data, 2)
        batch = train_data[:, i:min(i+batch_size-1, size(train_data, 2))]
        targets = train_targets[i:min(i+batch_size-1, size(train_targets, 1))]

        grads = gradient(() -> loss(batch, targets), params(model))
        Flux.Optimise.update!(optimizer, params(model), grads)
    end

    if (epoch) % 10 == 0
        @show epoch, loss(train_data, train_targets)
    end
end

# Evaluation
outputs = model(test_data)
predicted = outputs |> cpu
mse = mean((predicted .- test_targets |> cpu) .^ 2)
println("Mean Squared Error (MSE): $mse")