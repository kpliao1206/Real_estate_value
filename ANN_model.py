import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchsummary import summary

def linear_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Mish(),
        nn.Dropout(0.05)
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