import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchsummary import summary

def linear_block(in_f, out_f, drop=0.025):
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
        self.hidden1 = linear_block(in_features, 24)
        self.hidden2 = linear_block(24, 48)
        self.hidden3 = linear_block(48, 48)
        self.hidden4 = linear_block(48, 48)
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
        self.hidden1 = linear_block(in_features, 24)
        self.hidden2 = linear_block(24, 60)
        self.hidden3 = linear_block(60, 60)
        self.hidden4 = linear_block(60, 60)
        self.out = nn.Linear(60, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
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


class Features3_NN(torch.nn.Module):
    def __init__(self, in_features=3):
        super(Features3_NN, self).__init__()
        self.hidden1 = linear_block(in_features, 6)
        self.hidden2 = linear_block(6, 12)
        self.hidden3 = linear_block(12, 12)
        self.out = nn.Linear(12, 1)
        

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        output = self.out(x)
        
        return output


if __name__ == '__main__':
    model = Features12_NN(12).cuda()
    summary(model, input_size=(1, 12))