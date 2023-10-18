import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

class Features12_dataset(Dataset):
    def __init__(self, X, y, train=True, random_state=12):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        self.scaler_y = FunctionTransformer(np.log1p, np.expm1) # log(x+1), 使資料更接近常態分佈 inverse_func: np.expm1

        if train:
            # scaler_X = StandardScaler()
            scaler_X = Pipeline([
                ('MinMax', MinMaxScaler()),
                ('log1p', FunctionTransformer(np.log1p, np.expm1))     
            ])
            scaler_X.fit(X_train)
            X_scaled_train = scaler_X.transform(X_train)
            self.input = X_scaled_train

            # self.scaler_y = MinMaxScaler()
            # self.scaler_y.fit(y_train)
            self.target = self.scaler_y.transform(y_train)
        else:
            # scaler_X = StandardScaler()
            scaler_X = Pipeline([
                ('MinMax', MinMaxScaler()),
                ('log1p', FunctionTransformer(np.log1p, np.expm1))     
            ])
            scaler_X.fit(X_test)
            X_scaled_test = scaler_X.transform(X_test)
            self.input = X_scaled_test

            # self.scaler_y = MinMaxScaler()
            # self.scaler_y.fit(y_test)
            self.target = self.scaler_y.transform(y_test)
        self.input = torch.from_numpy(self.input)
        self.target = torch.from_numpy(self.target)

    def __getitem__(self, index):
        return self.input[index].type(torch.FloatTensor), self.target[index].type(torch.FloatTensor)

    def __len__(self):
        return self.input.shape[0]
    
    def get_y_scaler(self):
        return self.scaler_y
