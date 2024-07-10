import torch
import random
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import config as cfg


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MinMaxNorm:
    """ 缩放数据至[0, 1]区间"""

    def __init__(self):
        self.low = np.array(0)
        self.high = np.array(0)

    def fit(self, x):
        self.low = np.min(x, axis=0)
        self.high = np.max(x, axis=0)

    def transform(self, x, pos=-1):
        if pos == -1:
            x = 1.0 * (x - self.low) / (self.high - self.low)
        else:
            x = 1.0 * (x - self.low[pos]) / (self.high[pos] - self.low[pos])
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x, pos=None):
        if pos is None:
            x = x * (self.high - self.low) + self.low
        else:
            x = x * (self.high[pos] - self.low[pos]) + self.low[pos]
        return x


class StandardNorm:
    """ 缩放数据至0均值1方差"""

    def __init__(self):
        self.mean = np.array(0)
        self.std = np.array(0)

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x, pos=None):
        if pos is None:
            x = 1.0 * (x - self.mean) / self.std
        else:
            x = 1.0 * (x - self.mean[pos]) / self.std[pos]
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x, pos=None, keep_positive=True):
        if pos is None:
            x = x * self.std + self.mean
            if keep_positive:
                x = np.maximum(x, np.zeros_like(x))
        else:
            x = x * self.std[pos] + self.mean[pos]
            if keep_positive:
                x = np.maximum(x, np.zeros_like(x))
        return x


class EarlyStopping:
    """
    早停并保存模型
    """

    def __init__(self, logger, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), path + 'checkpoint.pth')
        self.val_loss_min = val_loss


def plot_fig(y_test, y_pred, folder_name, prefix=''):
    fig, ax = plt.subplots(figsize=(20, 5))
    line1, = ax.plot(y_test, label='Actual')
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    line2, = ax.plot(y_pred, dashes=[6, 2], label='Forecast')
    # ax.set_ylim(-10, 20)
    ax.legend()
    plt.savefig(f'./experiment/main_exp/{cfg.exp_type}/{folder_name}/{prefix}result.png')
    plt.close()


def calculate_metric(trues, preds, logger):
    """
    计算实验结果
    :param trues: 真实值
    :param preds: 预测值
    :param logger: 保存的文件夹名称
    :param args: 参数
    """
    test_r2 = r2_score(trues, preds)
    test_mse = (np.square(trues - preds)).mean()
    test_rmse = np.sqrt(test_mse)
    test_mae = (np.abs(trues - preds)).mean()
    logger.info(
        f"Test MSE:  {test_mse:.4f}   |  Test RMSE:  {test_rmse:.4f}   |  Test MAE:  {test_mae:.4f}   |  Test R2:  {test_r2:.4f}|")
    return test_mse, test_rmse, test_mae, test_r2


def save_loss(train_loss, valid_loss, folder_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    line1, = ax.plot(train_loss, label='train loss')
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    line2, = ax.plot(valid_loss, dashes=[6, 2], label='valid loss')
    ax.legend()
    plt.savefig(f'./experiment/main_exp/{cfg.exp_type}/{folder_name}/loss.png')


def save_args():
    """
    保存实验参数信息，创建文件夹，文件夹名称为当前时间

    :return: 文件夹名称
    """
    if not os.path.exists(f'./experiment/main_exp/{cfg.exp_type}'):
        os.makedirs(f'./experiment/main_exp/{cfg.exp_type}')
    args_dict = cfg.__dict__
    now_time = datetime.datetime.now().strftime('%F_%H-%M-%S')
    os.makedirs(f'./experiment/main_exp/{cfg.exp_type}/{now_time}')
    with open(f'./experiment/main_exp/{cfg.exp_type}/{now_time}/args.txt', 'w') as f:
        f.writelines('------------------ args ------------------' + '\n')
        for each_arg, value in args_dict.items():
            if not each_arg.startswith('__'):
                f.writelines(each_arg + ' : ' + str(value) + '\n')
    return now_time
