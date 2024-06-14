from copy import deepcopy as dc
import pandas as pd
import torch
from torch.utils.data import Dataset

def prepare_dataframe_for_LSTM(df, window_size):
    df = dc(df)
    # df.set_index("Date", inplace=True)
    for i in range(1, window_size+1):
        df[f'Spread(t-{i})'] = df['Spread'].shift(i)

    df.dropna(inplace=True)
    return df

def split_train_test(X, y, train_proportion):
    split_index = int(len(X) * train_proportion)
    # X_train = torch.tensor(X[:split_index]).float()
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test

def reshape_data(X, y, window_size):
    X = X.reshape((-1, window_size, 1))
    y = y.reshape((-1, 1))

    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    return X, y


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]