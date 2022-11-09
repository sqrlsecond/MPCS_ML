import numpy as np
import torch
from torch.utils.data import Dataset

class MpcsDataset(Dataset):
    def __init__(self, mpcs, labels):
        self.mpcs = mpcs
        self.labels = labels

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):

        label = self.labels[idx]
        labels_tensor = torch.zeros([3], dtype=torch.float32)
        labels_tensor[label] = 1.0
        mpc_tensor = torch.from_numpy(self.mpcs[idx])
        return mpc_tensor, labels_tensor