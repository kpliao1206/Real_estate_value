import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_absolute_percentage_error, mean_squared_error


def train_model(model,
                train_loader, test_loader,
                num_epochs=300,
                lr=1e-2,
                weight_decay=1e-3,
                factor=0.3,
                min_lr=1e-6,
                verbose=True,
                patience=20,
                threshold=0.01
                ):
    
    model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # Adam with weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=verbose, min_lr=min_lr, threshold=threshold) # 依照cosine週期衰減
    criterion = nn.MSELoss()

    start = time.time()

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    train_mapes = []
    valid_mapes = []
    for epoch in range(num_epochs):

        ##### Training loop #####
        model.train() # prep model for training
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = torch.log(targets.cuda())

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets) # use log
            mape = mean_absolute_percentage_error(torch.exp(outputs.view(-1, 1)), torch.exp(targets.view(-1, 1))) # not use log

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_mapes.append(mape.item())

        ##### Validation loop #####
        model.eval() # prep model for evaluation
        for inputs, targets in test_loader:
            inputs = inputs.cuda()
            targets = torch.log(targets.cuda())

            outputs = model(inputs)
            test_loss = criterion(outputs, targets) # use log
            test_mape = mean_absolute_percentage_error(torch.exp(outputs.view(-1, 1)), torch.exp(targets.view(-1, 1))) # not use log
            valid_losses.append(test_loss.item())
            valid_mapes.append(test_mape.item())

        # update lr
        scheduler.step(test_loss)
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_mape = np.average(train_mapes) * 100
        valid_mape = np.average(valid_mapes) * 100

        print(f'[Epoch {epoch+1}/{num_epochs}] train_loss: {train_loss:.3f}, test_loss: {valid_loss:.3f} | train_mape: {train_mape:.2f}, test_mape: {valid_mape:.2f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # save state_dict
        # 創建資料夾
        path = "./State_dicts"
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(model.state_dict(), path+'/epoch'+str(epoch+1)+'.pt')


    end = time.time()
    print(f'Training is end. Total trainig time: {(end-start)/60:.1f} minutes')
    print(f'Min test_loss is at epoch{np.argmin(avg_valid_losses)+1}')

    return  model, avg_train_losses, avg_valid_losses