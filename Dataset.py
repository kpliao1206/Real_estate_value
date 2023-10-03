import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Features12_dataset(Dataset):
    def __init__(self, X, y, train=True, random_state=12):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        if train:
            self.input = X_train
            self.target = y_train
        else:
            self.input = X_test
            self.target = y_test
        self.input = torch.from_numpy(self.input)
        self.target = torch.from_numpy(self.target)

    def __getitem__(self, index):
        return self.input[index].type(torch.FloatTensor), self.target[index].type(torch.FloatTensor)

    def __len__(self):
        return self.input.shape[0]