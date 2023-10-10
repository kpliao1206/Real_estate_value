import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from Dataset import Features12_dataset


def train_model(model,
                train_loader, test_loader,
                y_scaler_train, y_scaler_test,
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=verbose, min_lr=min_lr, threshold=threshold)
    # scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    criterion = nn.HuberLoss(delta=1)

    start = time.time()

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    train_mapes = []
    valid_mapes = []
    avg_train_mapes = []
    avg_valid_mapes = []
    for epoch in range(num_epochs):

        ##### Training loop #####
        model.train() # prep model for training
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda() 

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets) # use log
            mape = mean_absolute_percentage_error(y_scaler_train.inverse_transform(targets.detach().cpu().numpy()).reshape(-1, 1), y_scaler_train.inverse_transform(outputs.detach().cpu().numpy()).reshape(-1, 1))

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
            targets = targets.cuda()

            outputs = model(inputs)
            test_loss = criterion(outputs, targets) # use log
            test_mape = mean_absolute_percentage_error(y_scaler_test.inverse_transform(targets.detach().cpu().numpy()).reshape(-1, 1), y_scaler_test.inverse_transform(outputs.detach().cpu().numpy()).reshape(-1, 1))

            valid_losses.append(test_loss.item())
            valid_mapes.append(test_mape.item())

        # update lr
        scheduler.step(test_mape)
        # scheduler.step()
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_mape = np.average(train_mapes) * 100
        valid_mape = np.average(valid_mapes) * 100

        avg_train_mapes.append(train_mape)
        avg_valid_mapes.append(valid_mape)

        print(f'[Epoch {epoch+1}/{num_epochs}] train_loss: {train_loss:.6f}, test_loss: {valid_loss:.6f} | train_mape: {train_mape:.4f}, test_mape: {valid_mape:.4f}')

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        train_mapes = []
        valid_mapes = []

        # save state_dict
        # 創建資料夾
        path = "./State_dicts"
        if not os.path.isdir(path):
            os.makedirs(path)
        torch.save(model.state_dict(), path+'/epoch'+str(epoch+1)+'.pt')


    end = time.time()
    print(f'Training is end. Total trainig time: {(end-start)/60:.1f} minutes')
    print(f'Min test_loss is at epoch{np.argmin(avg_valid_losses)+1}')
    print(f'Min test_mape is at epoch{np.argmin(avg_valid_mapes)+1} ({np.min(avg_valid_mapes)})')

    return  model, avg_train_losses, avg_valid_losses