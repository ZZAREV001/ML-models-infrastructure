import torch
import pandas as pd

df_open_interest = pd.read_csv(
    '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/open_interest.csv',
    parse_dates=['TimeStamp'])
df_bids = pd.read_csv(
    '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/bybit_bids.csv')
df_asks = pd.read_csv(
    '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/bybit_asks.csv')
df_trades = pd.read_csv(
    '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/recentTrades.csv')

# Convert 'time' column to datetime and set it as the index of df_trades
df_trades['time'] = pd.to_datetime(df_trades['time'])
df_trades.set_index('time', inplace=True)

# Resample open interest data to 5-minute intervals
df_open_interest_resampled = df_open_interest.resample('5min', on='TimeStamp').mean()

# Convert the index of df_open_interest_resampled to timezone-naive
df_open_interest_resampled.index = df_open_interest_resampled.index.tz_localize(None)

# Resample trade data to 5-minute intervals
df_trades_resampled = df_trades.resample('5min').agg({
    'price': ['first', 'max', 'min', 'last'],
    'size': 'sum'
})
df_trades_resampled.columns = ['open', 'high', 'low', 'close', 'volume']

# Merge resampled open interest and trade data using outer join
df_merged = pd.merge(df_trades_resampled, df_open_interest_resampled, left_index=True, right_index=True, how='outer')

# Fill missing values in the merged DataFrame
df_merged = df_merged.fillna(method='ffill')

# Save the merged DataFrame to a CSV file
output_file = 'global_analysis/processed/merged_data.csv'
df_merged.to_csv(output_file)

class CryptoDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

if not df_merged.empty:
    # Convert the merged DataFrame to PyTorch tensors
    data_tensor = torch.tensor(df_merged.values, dtype=torch.float32)

    # Create the PyTorch dataset
    dataset = CryptoDataset(data_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
else:
    print("The merged DataFrame is empty. Unable to create dataset and data loaders.")