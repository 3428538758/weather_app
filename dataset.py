# fire_damage_pred/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class FireDataset(Dataset):
    def __init__(self, feature_path="features.npy", label_path="labels.npy"):
        self.X = np.load(feature_path)   # (N, T, C, H, W)
        self.y = np.load(label_path)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
