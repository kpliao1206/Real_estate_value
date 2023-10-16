import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchsummary import summary

def linear_block(in_f, out_f, drop=0.):
    # weight initialize
    linear_layer = nn.Linear(in_f, out_f)
    torch.nn.init.xavier_uniform_(linear_layer.weight)
    torch.nn.init.zeros_(linear_layer.bias)

    return nn.Sequential(
        linear_layer,
        nn.Mish(),
        nn.Dropout(drop)
    )

class Features12_NN(torch.nn.Module):
    def __init__(self, in_features=12):
        super(Features12_NN, self).__init__()
        self.hidden1 = linear_block(in_features, 24, drop=0.01)
        self.hidden2 = linear_block(24, 48, drop=0.01)
        self.hidden3 = linear_block(48, 48, drop=0.01)
        self.hidden4 = linear_block(48, 48, drop=0.01)
        self.out = nn.Linear(48, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        output = self.out(x)
        
        return output


class Features12_NN2(torch.nn.Module):
    def __init__(self, in_features=12):
        super(Features12_NN2, self).__init__()
        self.hidden1 = linear_block(in_features, 24, drop=0.01)
        self.hidden2 = linear_block(24, 48, drop=0.01)
        self.hidden3 = linear_block(48, 48, drop=0.01)
        self.hidden4 = linear_block(48, 24, drop=0.01)
        self.hidden5 = linear_block(24, 24, drop=0.01)
        self.out = nn.Linear(24, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        output = self.out(x)
        
        return output


class Features13_NN(torch.nn.Module):
    def __init__(self, in_features=13):
        super(Features13_NN, self).__init__()
        self.hidden1 = linear_block(in_features, 24, drop=0.05)
        self.hidden2 = linear_block(24, 48, drop=0.05)
        self.hidden3 = linear_block(48, 48, drop=0.05)
        self.hidden4 = linear_block(48, 48, drop=0.05)
        self.out = nn.Linear(48, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        output = self.out(x)
        
        return output
    

class Features16_NN(torch.nn.Module):
    def __init__(self, in_features=16):
        super(Features16_NN, self).__init__()
        self.hidden1 = linear_block(in_features, 24, drop=0.1)
        self.hidden2 = linear_block(24, 48, drop=0.1)
        self.hidden3 = linear_block(48, 48, drop=0.1)
        self.hidden4 = linear_block(48, 48, drop=0.1)
        self.out = nn.Linear(48, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        output = self.out(x)
        
        return output


class Features16_NN2(torch.nn.Module):
    def __init__(self, in_features=16):
        super(Features16_NN2, self).__init__()
        self.hidden1 = linear_block(in_features, 32, drop=0.1)
        self.hidden2 = linear_block(32, 64, drop=0.1)
        self.hidden3 = linear_block(64, 64, drop=0.1)
        self.hidden4 = linear_block(64, 64, drop=0.1)
        self.out = nn.Linear(64, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        output = self.out(x)
        
        return output


class Features2_NN(torch.nn.Module):
    def __init__(self, in_features=2):
        super(Features2_NN, self).__init__()
        self.hidden1 = linear_block(in_features, 8)
        self.hidden2 = linear_block(8, 16)
        self.hidden3 = linear_block(16, 32)
        self.hidden4 = linear_block(32, 16)
        self.out = nn.Linear(16, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        output = self.out(x)
        
        return output


if __name__ == '__main__':
    model = Features12_NN(12).cuda()
    summary(model, input_size=(1, 12))