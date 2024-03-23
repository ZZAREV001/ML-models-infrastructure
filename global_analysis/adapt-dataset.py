import pandas as pd
import torch


def load_and_preprocess_csv(filepath, datetime_col, resample=None, agg_dict=None):
    df = pd.read_csv(filepath)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.dropna(inplace=True)
    df.set_index(datetime_col, inplace=True)
    if resample and agg_dict:
        df = df.resample(resample).agg(agg_dict)
    return df


df_trades = load_and_preprocess_csv(
    '/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/recentTrades.csv',
    'time', '5min', {'price': 'mean', 'size': 'sum'})
df_open_interest = load_and_preprocess_csv('/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/open_interest.csv',
                                           'TimeStamp')

# Convert the index of df_open_interest to timezone-naive
df_open_interest.index = df_open_interest.index.tz_localize(None)

df_asks = pd.read_csv('/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/bybit_bids.csv')
df_bids = pd.read_csv('/Users/GoldenEagle/Desktop/Divers/Dossier-cours-IT/Cours-JavaScript/Project-Derivative-spot/bybit_asks.csv')

df_merged = df_trades.join(df_open_interest, how='outer')
merged_tensor = torch.tensor(df_merged.values, dtype=torch.float32)
asks_tensor = torch.tensor(df_asks.values, dtype=torch.float32)
bids_tensor = torch.tensor(df_bids.values, dtype=torch.float32)

# Determine the maximum number of rows among the tensors
max_rows = max(merged_tensor.shape[0], asks_tensor.shape[0], bids_tensor.shape[0])

# Pad the tensors with zeros to match the maximum number of rows
merged_tensor = torch.nn.functional.pad(merged_tensor, (0, 0, 0, max_rows - merged_tensor.shape[0]))
asks_tensor = torch.nn.functional.pad(asks_tensor, (0, 0, 0, max_rows - asks_tensor.shape[0]))
bids_tensor = torch.nn.functional.pad(bids_tensor, (0, 0, 0, max_rows - bids_tensor.shape[0]))

final_tensor = torch.cat([merged_tensor, asks_tensor, bids_tensor], dim=1)

# Save the tensors to a file
torch.save(final_tensor, 'final_tensor.pt')