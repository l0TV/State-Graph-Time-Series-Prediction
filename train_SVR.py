import os.path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics
from utils import set_seed
from dataloader import get_ml_loader
import scipy.io as io


def single_SVR_train(is_train=True):
    set_seed(1022)
    save_path = f'./experiment/main_exp/SVR/'
    train_x, train_y, test_x, test_y, mmn = get_ml_loader()

    if is_train:
        estimator = SVR(C=1, epsilon=0.3, kernel='rbf')
        estimator.fit(train_x, train_y)
        # 保存实验结果
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        joblib.dump(estimator, save_path + 'SVR.pkl')

    svr = joblib.load(save_path + 'SVR.pkl')
    kinds = os.listdir('./turned_dataset/LTE_Dataset')
    f = open(save_path + 'exp.txt', 'w+')
    for kind in kinds:
        f.write('-------' + kind + '-------\n')
        y_pred = mmn.inverse_transform(svr.predict(test_x[kind]), -1)
        test_y_ = mmn.inverse_transform(test_y[kind], -1)
        f.write('RMSE:' + str(np.sqrt(metrics.mean_squared_error(test_y_, y_pred))) + '|\t')
        f.write('MAE:' + str(metrics.mean_absolute_error(test_y_, y_pred)) + '|\t')
        f.write('R2:' + str(metrics.r2_score(test_y_, y_pred)) + '\n')
        result_pd = pd.DataFrame({'preds': y_pred.flatten(), 'trues': test_y_.flatten()})
        result_pd.to_csv(save_path + f'{kind}_result.csv', index=False, sep=',')
        io.savemat(save_path + f'{kind}_result.mat', {'preds': y_pred, 'trues': test_y_})

        # 画图保存
        fig, ax = plt.subplots(figsize=(20, 5))
        line1, = ax.plot(test_y_, label='Actual')
        line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
        line2, = ax.plot(y_pred, dashes=[6, 2], label='Forecast')
        # ax.set_ylim(-10, 20)
        ax.legend()
        plt.savefig(save_path + f'{kind}result.png')
        plt.close()


single_SVR_train(False)
