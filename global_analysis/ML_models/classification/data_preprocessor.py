import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load_and_preprocess_data(self):
        # Load and preprocess the data from each CSV file, excluding historical volatility
        open_interest_data = pd.read_csv(self.file_paths['open_interest'])
        bid_data = pd.read_csv(self.file_paths['bid_data'], header=None, names=['bid_price', 'bid_size'])
        ask_data = pd.read_csv(self.file_paths['ask_data'], header=None, names=['ask_price', 'ask_size'])
        recent_trades_data = pd.read_csv(self.file_paths['recent_trades'])

        # Convert timestamps to datetime format
        open_interest_data['TimeStamp'] = pd.to_datetime(open_interest_data['TimeStamp'], utc=True)
        recent_trades_data['time'] = pd.to_datetime(recent_trades_data['time'], utc=True)
        bid_data.index = pd.to_datetime(bid_data.index, utc=True)
        ask_data.index = pd.to_datetime(ask_data.index, utc=True)

        # Sort the DataFrames by their respective timestamp columns
        open_interest_data.sort_values('TimeStamp', inplace=True)
        recent_trades_data.sort_values('time', inplace=True)

        # Start merging with the dataset that has timestamps (choose one as a base, e.g., open_interest_data)
        merged_data = open_interest_data.copy()
        merged_data = pd.merge_asof(merged_data, bid_data, left_on='TimeStamp', right_index=True, direction='nearest')
        merged_data = pd.merge_asof(merged_data, ask_data, left_on='TimeStamp', right_index=True, direction='nearest')
        merged_data = pd.merge_asof(merged_data, recent_trades_data, left_on='TimeStamp', right_on='time', direction='nearest')

        print("Merged Dataset:")
        print(merged_data.head())
        print("Merged Dataset Shape:", merged_data.shape)

        # Select relevant features from the merged dataset, excluding 'Value'
        features = ['OpenInterest', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'price', 'size']

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
        y = []  # Adjust how you determine 'y' based on your new target
        # Ensure 'y' is populated based on the new target decision
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
