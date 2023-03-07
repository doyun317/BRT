import os
import pandas as pd
import torch
import numpy
from torch.utils.data import Dataset, DataLoader

class CoinDataset(Dataset):
    def __init__(self, file_name):
        self.raw_data = pd.read_csv(file_name)
        self.state_data = torch.from_numpy(self.raw_data.iloc[:, 1:].values).float()
        self.price_data = torch.from_numpy(self.raw_data["Price"].values).float()

    def __len__(self):
        return len(self.state_data)

    def __getitem__(self, idx):
        return self.state_data[idx].unsqueeze(0), self.price_data[idx]

