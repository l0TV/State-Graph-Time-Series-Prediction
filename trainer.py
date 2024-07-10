import logging
import time
import pandas as pd
from tqdm import tqdm
import scipy.io as io
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
from model.state_graph import attention_select
from model.loss import WeightedMSE
import config as cfg


class Experiment:
    def __init__(self, model, data_loader, device):
        self.data_loader = data_loader
        self.model = model
        self.device = device

        if cfg.train:
            #  创建实验文件夹
            if not os.path.exists('./experiment/main_exp'):
                os.makedirs('./experiment/main_exp')
            self.folder_name = save_args()
            self.logger = self.get_logger(f'./experiment/main_exp/{cfg.exp_type}/{self.folder_name}/exp.log')
        else:
            self.folder_name = cfg.load_model_path
        self.path = f'./experiment/main_exp/{cfg.exp_type}/{self.folder_name}/'

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
        return model_optim

    def _select_scheduler(self, optimizer):
        if cfg.lr_scheduler == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg.miletones,
                                                       gamma=cfg.gamma, verbose=True)
        elif cfg.lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=cfg.gamma,
                                                             patience=cfg.patience - 1, verbose=True)
        else:
            raise RuntimeError('未知的学习率规划器。')
        return scheduler

    def _select_criterion(self):
        loss = {'MAE': nn.L1Loss(), 'MSE': nn.MSELoss(), 'CrossEntropy': nn.CrossEntropyLoss(),
                'WeightedMSE': WeightedMSE(cfg.loss_weight)}
        criterion = loss[cfg.loss]
        return criterion

    def get_logger(self, filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    def train(self):
        folder_name = self.folder_name
        path = self.path

        train_loader = self.data_loader.train
        val_loader = self.data_loader.val
        # test_loader = self.data_loader.test
        device = self.device

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        criterion = self._select_criterion()

        self.logger.info('-----Start training-----')

        early_stopping = EarlyStopping(logger=self.logger, patience=cfg.patience, verbose=False)

        train_loss_list = []
        val_loss_list = []
        for epoch in range(cfg.epoch):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for data in tqdm(train_loader):
                if not cfg.model in ['GRU', 'LSTM']:
                    graph_list = data['graph']
                if cfg.use_label_bins:
                    ground_truth = data['label'].type(torch.int64).to(device)
                else:
                    # ground_truth = data['label'].type(torch.float32).to(device)
                    ground_truth = data['full_label'].type(torch.float32).to(device)
                model_optim.zero_grad()

                if cfg.use_state_attention:
                    series = data['series'].type(torch.float32).to(device)
                    output, atten = self.model(graph_list, series)
                elif cfg.model == 'GRU' or cfg.model == 'LSTM':
                    series = data['series'].type(torch.float32).to(device)
                    output = self.model(series)
                elif cfg.use_series or cfg.use_parallel_encoder:
                    series = data['series'].type(torch.float32).to(device)
                    # bw_bins = self.data_loader.scaler.transform(self.data_loader.bw_bins.reshape(-1, 1))
                    bw_bins = self.data_loader.bw_bins
                    if cfg.use_multi_loss:
                        output, prediction_label_bins = self.model(graph_list, series, bw_bins)
                    else:
                        output = self.model(graph_list, series, bw_bins)
                else:
                    output = self.model(graph_list)

                if cfg.use_multi_loss:
                    label_bins = torch.from_numpy(attention_select(bw_bins, ground_truth)).type(torch.LongTensor).to(
                        device)
                    loss1 = criterion(output, ground_truth)
                    loss2 = F.cross_entropy(prediction_label_bins, label_bins)
                    loss = cfg.loss_weight1 * loss1 + cfg.loss_weight2 * loss2
                else:
                    loss = criterion(output, ground_truth)

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(val_loader, criterion, device)
            # test_loss = self.vali(test_loader, criterion, device)

            train_loss_list.append(train_loss)
            val_loss_list.append(vali_loss)

            if cfg.lr_scheduler == 'plateau':
                scheduler.step(vali_loss)
            else:
                scheduler.step()

            self.logger.info(
                'Epoch:[{}/{}]\t Train loss={:.7f}\t Vali Loss={:.7f}\t Cost time={:.1f}'.format(
                    epoch + 1, cfg.epoch, train_loss, vali_loss, time.time() - epoch_time))

            # 保存训练过程中在验证集上的误差最小的模型参数，并检测训练过程中模型在验证集上的误差相比于最佳时的情况是否有所减少，
            # 若连续patience次没有减少，则提前停止训练
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping!")
                break
        early_stopping.save_checkpoint(vali_loss, self.model, path + 'last_model/')
        train_loss_np = np.array(train_loss_list)
        valid_loss_np = np.array(val_loss_list)
        save_loss(train_loss_np, valid_loss_np, folder_name)  # 保存误差的变化曲线

        best_model_path = path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.logger.info('Finish training!')

    def vali(self, vali_loader, criterion, device):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(vali_loader):
                if cfg.model not in ['GRU', 'LSTM']:
                    graph_list = data['graph']
                if cfg.use_label_bins:
                    ground_truth = data['label'].type(torch.int64).to(device)
                else:
                    ground_truth = data['full_label'].type(torch.float32).to(device)
                    # ground_truth = data['label'].type(torch.float32).to(device)

                if cfg.use_state_attention:
                    series = data['series'].type(torch.float32).to(device)
                    outputs, atten = self.model(graph_list, series)
                elif cfg.model == 'GRU' or cfg.model == 'LSTM':
                    series = data['series'].type(torch.float32).to(device)
                    output = self.model(series)
                elif cfg.use_series or cfg.use_parallel_encoder:
                    series = data['series'].type(torch.float32).to(device)
                    bw_bins = self.data_loader.bw_bins
                    if cfg.use_multi_loss:
                        output, prediction_label_bins = self.model(graph_list, series, bw_bins)
                    else:
                        output = self.model(graph_list, series, bw_bins)
                else:
                    outputs = self.model(graph_list)

                if cfg.use_multi_loss:
                    label_bins = torch.from_numpy(attention_select(bw_bins, ground_truth)).type(torch.LongTensor)
                    loss1 = criterion(output.detach().cpu(), ground_truth.detach().cpu())
                    loss2 = F.cross_entropy(prediction_label_bins.detach().cpu(), label_bins.detach().cpu())
                    loss = cfg.loss_weight1 * loss1 + cfg.loss_weight2 * loss2
                else:
                    loss = criterion(output, ground_truth)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, is_last_model=False):
        if not is_last_model:
            folder_name = self.folder_name
        else:
            folder_name = self.folder_name + '/last_model'
        test_loader = self.data_loader.test
        device = self.device

        if not cfg.train:
            self.logger = self.get_logger(f'./experiment/main_exp/{cfg.exp_type}/{folder_name}/test.log')
            print('loading model...')
        else:
            self.logger.info('loading model...')

        best_model_path = f'./experiment/main_exp/{cfg.exp_type}/{folder_name}/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))

        kinds = os.listdir('./turned_dataset/LTE_Dataset')
        preds = {}
        trues = {}

        self.model.eval()
        with torch.no_grad():
            for kind, loader in test_loader.items():
                preds[kind] = []
                trues[kind] = []
                for data in tqdm(loader):
                    if cfg.model not in ['GRU', 'LSTM']:
                        graph_list = data['graph']
                    if cfg.use_label_bins:
                        ground_truth = data['label'].type(torch.int64).to(device)
                    else:
                        ground_truth = data['label'].type(torch.float32).to(device)

                    if cfg.use_state_attention:
                        series = data['series'].type(torch.float32).to(device)
                        outputs, atten = self.model(graph_list, series)
                    elif cfg.model == 'GRU' or cfg.model == 'LSTM':
                        series = data['series'].type(torch.float32).to(device)
                        outputs = self.model(series)
                    elif cfg.use_series or cfg.use_parallel_encoder:
                        series = data['series'].type(torch.float32).to(device)
                        bw_bins = self.data_loader.bw_bins
                        if cfg.use_multi_loss:
                            outputs, prediction_label_bins = self.model(graph_list, series, bw_bins)
                        else:
                            outputs = self.model(graph_list, series, bw_bins)
                    else:
                        outputs = self.model(graph_list)

                    if cfg.use_label_bins:
                        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                    else:
                        pred = outputs.detach().cpu().numpy()
                    pred = pred[:, -1]
                    true = ground_truth.detach().cpu().numpy()

                    preds[kind].append(pred)
                    trues[kind].append(true)

        for i, kind in enumerate(kinds):
            pred_ = np.concatenate(preds[kind], axis=0)
            true_ = np.concatenate(trues[kind], axis=0)

            y_scaler = self.data_loader.scaler
            pred_ = y_scaler.inverse_transform(pred_.reshape(-1, 1), -1)
            true_ = y_scaler.inverse_transform(true_.reshape(-1, 1), -1)
            save_path = f'./experiment/main_exp/{cfg.exp_type}/{folder_name}/'
            result_pd = pd.DataFrame({'preds': pred_.flatten(), 'trues': true_.flatten()})
            result_pd.to_csv(save_path + kind + '_result.csv', index=False, sep=',')
            # io.savemat(save_path + 'result.mat', {'preds': preds, 'trues': trues, 'atten': attens})
            io.savemat(save_path + kind + '_result.mat', {'preds': pred_, 'trues': true_})

            plot_fig(true_, pred_, folder_name, kind + '_')
            calculate_metric(true_, pred_, self.logger)
