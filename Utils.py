import numpy as np
import matplotlib.pyplot as plt
import os


def loss_plot(avg_train_losses, avg_test_losses):
    plt.figure()
    plt.plot(avg_train_losses, 'r-', label='train')
    plt.plot(avg_test_losses, 'b-', label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('recorded loss')
    plt.legend()
    plt.show()


def true_pred_plot(y_pred, y_true, title='', llim=0, rlim=14):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(llim, rlim)
    ax.set_ylim(llim, rlim)
    ax.plot(y_pred, y_true, 'ro', markersize=5)
    ax.plot([llim, rlim], [llim, rlim], 'k--', linewidth=1.5)
    ax.set_xlabel('predicted')
    ax.set_ylabel('ground truth')
    # ax.legend()
    plt.show()


def model_pred(model, data_loader, y_scaler):
    outputs_all = []
    targets_all = []
    model.eval() # prep model for evaluation
    for inputs, targets in data_loader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        outputs_all.append(y_scaler.inverse_transform(outputs.cpu().detach().numpy()))
        targets_all.append(y_scaler.inverse_transform(targets.cpu().detach().numpy()))
    return np.array(outputs_all).ravel(), np.array(targets_all).ravel()


def save_with_unique_name(file_path):
    original_file_path = file_path
    file_name, file_extension = os.path.splitext(file_path)
    counter = 1

    while os.path.exists(file_path):
        file_path = f"{file_name}_{counter}{file_extension}"
        counter += 1

    # 此时 file_path 包含了一个唯一的文件名
    # 您可以将数据保存到 file_path 中
    # 例如：df.to_csv(file_path, index=False)

    return file_path