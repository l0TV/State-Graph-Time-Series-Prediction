import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import ruptures as rpt
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import config as cfg
from utils import MinMaxNorm, StandardNorm, set_seed


class LTEDataset(Dataset):
    def __init__(self, dataset, index_list, bw_bins):
        self.dataset = dataset
        self.index_list = index_list
        self.bw_bins = bw_bins

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        sample = prepare_graph(idx, self.dataset)
        return sample


def prepare_graph(idx, dataset):
    graphs = dataset['graph_list'][idx]
    label = dataset['label'][idx]
    full_label = dataset['full_label'][idx]
    dl_history_scaled_np = dataset['series'][idx]
    return {'graph': graphs, 'series': dl_history_scaled_np[-cfg.series_length:, :], 'label': label,
            'full_label': full_label}


def cpd(series):
    """
    变点检测
    """
    algo = rpt.KernelCPD(kernel="rbf", min_size=20).fit(series)
    result = algo.predict(n_bkps=cfg.num_time_step - 1)
    # 划分成num_time_step个子序列则需要num_time_step - 1个变点，
    # 实际返回的是长度为num_step的一个数组，因为额外包含了数组中最后一个元素的下标

    # fig, _ = rpt.display(series, [0], computed_chg_pts=result)
    # fig.show()
    return result


def get_endpoint(norm_obj):
    if cfg.norm_type == 'max-min':
        max_min_bins = np.array(cfg.bins)
        # 均匀划分状态的效果不好，因为会受到少数很大的值的干扰，可以改为不均匀划分，0~20000的区间可以划分多一点
        max_min_bins = max_min_bins.astype(float)
    elif cfg.norm_type == 'std':
        # max = norm_obj[0].transform(norm_obj[1].high[-1])
        # min = norm_obj[0].transform(norm_obj[1].low[-1])
        # max_min_bins = np.linspace(min-1e-6, max, cfg.num_state, endpoint=True)  # 注意这里应该是 min-1e-6
        max_min_bins = norm_obj[0].transform(norm_obj[1].inverse_transform(np.array(cfg.bins), -1), -1).astype(float)
    else:
        raise RuntimeError('未知的归一化类型。')
    return max_min_bins


def series_cut(original_series, change_point):
    series = []
    change_point = np.insert(change_point, 0, 0)
    # change_point = np.insert(change_point, len(change_point), len(original_series))
    for i, point in enumerate(change_point):
        if i < len(change_point) - 1:
            series.append(original_series[change_point[i]:change_point[i + 1]])
    return series


def generate_graph(series, bw_bins):
    weight_list = []

    bw_bins_inf = np.zeros([bw_bins.shape[0] + 1])
    bw_bins_inf[:-1] = bw_bins
    bw_bins_inf[-1] = np.inf

    for segment in series:
        weight = torch.zeros([cfg.num_state, cfg.num_state], dtype=torch.float)
        state = pd.cut(segment, bw_bins_inf, labels=False)
        # if True in np.isnan(state):
        #     print(state)
        for i in range(len(state)):
            if i < len(state) - 1:
                weight[state[i], state[i + 1]] += 1

        # 邻接矩阵每个状态的行归一化
        if cfg.adj_row_norm:
            row_sum = torch.sum(weight, dim=1).reshape(-1, 1)
            row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
            weight = torch.div(weight, row_sum)
        else:
            weight = torch.div(weight, len(state) - 1)
        weight = torch.flatten(weight)[:, None]
        weight_list.append(weight)

    return weight_list


def get_ml_loader():
    set_seed(cfg.seed)
    kinds = os.listdir('./turned_dataset/LTE_Dataset')
    kinds_length = {}.fromkeys(kinds, 0)
    all_data = []
    series = []
    label = []
    train_index = []
    test_index_dict = {}
    test_x = {}  # {}.fromkeys(kinds, [])的这种写法有问题，因为这样每个key所对应的实际上是内存中同一个位置处的列表
    test_y = {}
    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        for j in range(0, len(ls)):
            data_term = np.array(pd.read_csv('./turned_dataset/LTE_Dataset/' + i + '/' + ls[j]))[:, 1:]
            all_data.append(data_term)
    all_data = np.concatenate(all_data, axis=0)
    if cfg.norm_type == 'max-min':
        mmn = MinMaxNorm()  # 也可尝试-均值÷标准差的那种标准化
        mmn.fit(all_data)
    elif cfg.norm_type == 'std':
        mmn_ = [StandardNorm(), MinMaxNorm()]
        mmn_[0].fit(all_data)
        mmn_[1].fit(all_data)
        mmn = mmn_[0]
    else:
        raise RuntimeError('未知的归一化类型。')
    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        random.shuffle(ls)
        for j in range(0, len(ls)):
            data_ori = pd.read_csv('./turned_dataset/LTE_Dataset/' + i + '/' + ls[j])
            data_ori = np.array(data_ori)[:, 1:]
            data_ori = mmn.transform(data_ori)
            for k in range(cfg.window_size - 1, data_ori.shape[0] - 1, cfg.window_step):
                kinds_length[i] += 1
                series.append(data_ori[k - cfg.series_length + 1:k + 1, -1][None, :])
                label.append(data_ori[k + 1, -1])
    series = np.concatenate(series, axis=0)
    label = np.array(label)

    last_value = 0
    for i, num in enumerate(kinds_length.values()):
        m1 = last_value + int(num * cfg.train_ratio)
        m2 = last_value + int(num * (cfg.train_ratio + cfg.test_ratio))
        train_index.extend(list(range(last_value, m1)))
        test_index_dict[kinds[i]] = list(range(m1, m2))
        last_value += num

    train_x = series[train_index, :]
    train_y = label[train_index]
    for i in kinds:
        test_x[i] = series[test_index_dict[i], :]
        test_y[i] = label[test_index_dict[i]]
    return train_x, train_y, test_x, test_y, mmn


def get_feature_analysis_loader():
    set_seed(cfg.seed)
    kinds = os.listdir('./turned_dataset/LTE_Dataset')
    kinds_length = {}.fromkeys(kinds, 0)
    all_data = []
    series = []
    label = []
    train_index = []
    test_index_dict = {}
    test_x = {}  # {}.fromkeys(kinds, [])的这种写法有问题，因为这样每个key所对应的实际上是内存中同一个位置处的列表
    test_y = {}
    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        for j in range(0, len(ls)):
            data_term = np.array(pd.read_csv('./turned_dataset/LTE_Dataset/' + i + '/' + ls[j]))[:, 1:]
            all_data.append(data_term)
    all_data = np.concatenate(all_data, axis=0)
    if cfg.norm_type == 'max-min':
        mmn = MinMaxNorm()  # 也可尝试-均值÷标准差的那种标准化
        mmn.fit(all_data)
    elif cfg.norm_type == 'std':
        mmn_ = [StandardNorm(), MinMaxNorm()]
        mmn_[0].fit(all_data)
        mmn_[1].fit(all_data)
        mmn = mmn_[0]
    else:
        raise RuntimeError('未知的归一化类型。')
    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        random.shuffle(ls)
        for j in range(0, len(ls)):
            data_ori = pd.read_csv('./turned_dataset/LTE_Dataset/' + i + '/' + ls[j])
            data_ori = np.array(data_ori)[:, 1:]
            data_ori = mmn.transform(data_ori)
            for k in range(cfg.window_size - 1, data_ori.shape[0] - 1, cfg.window_step):
                kinds_length[i] += 1
                series.append(data_ori[k, :][None, :])
                label.append(data_ori[k + 1, -1])
    series = np.concatenate(series, axis=0)
    label = np.array(label)

    last_value = 0
    for i, num in enumerate(kinds_length.values()):
        m1 = last_value + int(num * cfg.train_ratio)
        m2 = last_value + int(num * (cfg.train_ratio + cfg.test_ratio))
        train_index.extend(list(range(last_value, m1)))
        test_index_dict[kinds[i]] = list(range(m1, m2))
        last_value += num
    # index_list = []
    # for i, num in enumerate(kinds_length.values()):
    #     num_list = list(range(last_value, last_value + num))
    #     random.shuffle(num_list)
    #     train_index_temp = num_list[0:int(num * cfg.train_ratio)]
    #     test_index_dict[kinds[i]] = num_list[int(num * cfg.train_ratio):int(num * (cfg.train_ratio + cfg.test_ratio))]
    #     vali_index_temp = num_list[int(num * (cfg.train_ratio + cfg.test_ratio)):]
    #     index_list.append([train_index_temp, vali_index_temp])
    #     last_value += num
    # train_index = index_list[0][0] + index_list[1][0]

    train_x = series[train_index, :]
    train_y = label[train_index]
    for i in kinds:
        test_x[i] = series[test_index_dict[i], :]
        test_y[i] = label[test_index_dict[i]]
    return train_x, train_y, test_x, test_y, mmn


def get_loader_v1():
    """
    该版本函数的训练集和测试集划分方法是以轨迹为单位的，确保滑动窗口不会出现相似的时间序列，
    能较好地反映模型的真实预测能力，但是无法保证训练集和测试集的大小比例严格符合预期
    """
    set_seed(cfg.seed)
    kinds = os.listdir('./turned_dataset/LTE_Dataset')
    all_data = []
    train_data = []
    test_data_dict = {}.fromkeys(kinds, [])
    test_loader_dict = {}
    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        for j in range(0, len(ls)):
            all_data.append(np.array(pd.read_csv('./turned_dataset/LTE_Dataset/' + i + '/' + ls[j]))[:, 1:])
    all_data = np.concatenate(all_data, axis=0)
    mmn = MinMaxNorm()
    mmn.fit(all_data)

    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        random.shuffle(ls)
        for j in range(0, len(ls)):
            data_ori = pd.read_csv('./turned_dataset/' + i + '/' + ls[j])
            data_ori = np.array(data_ori)[:, 1:]
            data_ori = mmn.transform(data_ori)
            if j < len(ls) * cfg.train_ratio:
                for k in range(cfg.window_size - 1, data_ori.shape[0] - 1, cfg.window_step):
                    train_data.append((data_ori[k - cfg.window_size + 1:k + 1, :], data_ori[k + 1, -1]))
            else:
                for k in range(cfg.window_size - 1, data_ori.shape[0] - 1, cfg.window_step):
                    test_data_dict[i].append((data_ori[k - cfg.window_size + 1:k + 1, :], data_ori[k + 1, -1]))
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=0, shuffle=True)
    for i in test_data_dict.keys():
        test_loader_dict[i] = DataLoader(test_data_dict[i], batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    return train_loader, test_loader_dict, mmn


def get_loader():
    """
    该版本函数的训练集和测试集划分方法也是以轨迹为单位的，能保证训练集和测试集的大小比例严格符合预期，
    但是在交界处会出现相似的时间序列，是一个折中的解决方案
    """
    set_seed(cfg.seed)
    kinds = os.listdir('./turned_dataset/LTE_Dataset')
    kinds_length = {}.fromkeys(kinds, 0)
    all_data = []
    series = []
    label = []
    full_label = []
    cp = []
    graph = []
    train_index = []
    vali_index = []
    test_dataset = {}
    test_index_dict = {}  # {}.fromkeys(kinds, [])的这种写法有问题，因为这样每个key所对应的实际上是内存中同一个位置处的列表
    test_loader_dict = {}
    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        for j in range(0, len(ls)):
            data_term = np.array(pd.read_csv('./turned_dataset/LTE_Dataset/' + i + '/' + ls[j]))[:, 1:]
            all_data.append(data_term)
    all_data = np.concatenate(all_data, axis=0)
    if cfg.norm_type == 'max-min':
        mmn = MinMaxNorm()  # 也可尝试-均值÷标准差的那种标准化
        mmn.fit(all_data)
        bw_bins = get_endpoint(mmn)
    elif cfg.norm_type == 'std':
        mmn_ = [StandardNorm(), MinMaxNorm()]
        mmn_[0].fit(all_data)
        mmn_[1].fit(all_data)
        bw_bins = get_endpoint(mmn_)
        mmn = mmn_[0]
    else:
        raise RuntimeError('未知的归一化类型。')
    for i in kinds:
        ls = os.listdir('./turned_dataset/LTE_Dataset/' + i)
        random.shuffle(ls)
        for j in range(0, len(ls)):
            data_ori = pd.read_csv('./turned_dataset/LTE_Dataset/' + i + '/' + ls[j])
            data_ori = np.array(data_ori)[:, 1:]
            data_ori = mmn.transform(data_ori)
            for k in range(cfg.window_size - 1, data_ori.shape[0] - 1, cfg.window_step):
                kinds_length[i] += 1
                series.append(data_ori[k - cfg.window_size + 1:k + 1, :])
                label.append(data_ori[k + 1, -1])
                full_label.append(data_ori[k + 1, :])
                bw_series = data_ori[k - cfg.window_size + 1:k + 1, -1]
                cp_ = cpd(bw_series)
                cp.append(cp_)  # 此处以带宽作为变点检测的依据，也可试试其它

    path = f'./turned_dataset/LTE_Dataset_Graph_Seed_{cfg.seed}_UnevenStates_' \
           f'{cfg.num_state}_NumSeries_{cfg.num_time_step}.pkl'
    if cfg.load_graph:
        # 从文件中加载图
        with open(path, 'rb') as f:
            graph = pickle.load(f)
    else:
        for i in tqdm(range(len(series))):
            graph_ = generate_graph(series_cut(series[i][:, -1], cp[i]), bw_bins)
            graph.append(graph_)
        with open(path, 'wb') as f:
            pickle.dump(graph, f)

    last_value = 0
    for i, num in enumerate(kinds_length.values()):
        m1 = last_value + int(num * cfg.train_ratio)
        m2 = last_value + int(num * (cfg.train_ratio + cfg.test_ratio))
        m3 = last_value + num
        train_index.extend(list(range(last_value, m1)))
        test_index_dict[kinds[i]] = list(range(m1, m2))
        vali_index.extend(list(range(m2, m3)))
        last_value += num

    dataset = {'series': series, 'label': label, 'full_label': full_label, 'cp': cp, 'graph_list': graph}
    # series : List(数据项总数){Ndarray(时间序列长度，特征数){前若干个时刻的特征}}
    # label  : List(数据项总数){归一化的流量真值}
    # full_label  : List(数据项总数){Ndarray(特征数){要预测的时刻的所有特征}}
    # cp     : List(数据项总数){List(变点数+1){当前序列中的变点下标}}
    # graph  : List(数据项总数){List(变点数+1){Tensor(状态数平方,1){展平后的各个子序列的状态转换图邻接矩阵}}}
    train_dataset = LTEDataset(dataset, train_index, bw_bins)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=True)
    vali_dataset = LTEDataset(dataset, vali_index, bw_bins)
    vali_loader = DataLoader(vali_dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    for i in kinds:
        test_dataset[i] = LTEDataset(dataset, test_index_dict[i], bw_bins)
        test_loader_dict[i] = DataLoader(test_dataset[i], batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    return train_loader, vali_loader, test_loader_dict, mmn, bw_bins


class Loader:
    def __init__(self):
        self.train, self.val, self.test, self.scaler, self.bw_bins = get_loader()


if __name__ == '__main__':
    get_ml_loader()
