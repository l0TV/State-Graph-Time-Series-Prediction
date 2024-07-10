import os
import numpy as np
import pandas as pd
import config as cfg


def expo_avg(numlist):
    s = 0
    for i in numlist:
        s += np.power(10, i / 10)
    return 10 * (-np.log10(len(numlist)) + np.log10(s))


kinds = os.listdir('./dataset')
# data = []
for i in kinds:
    ls = os.listdir('./dataset/' + i)
    save_path = './turned_dataset/LTE_Dataset/' + i
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for j in ls:
        data_ori = pd.read_csv('./dataset/' + i + '/' + j)
        data_ori.loc[data_ori['NetworkMode'] != 'LTE', ['RSRQ', 'SNR', 'CQI', 'RSSI']] = [cfg.RSRQ_fill, cfg.SNR_fill,
                                                                                          cfg.CQI_fill, cfg.RSSI_fill]
        # dataframe最好使用.loc[]而不是直接使用[]，因为后者不支持同时使用布尔索引和普通索引；无论是使用.loc[]还是[]，
        # 最好不要连续调用两次，否则最终返回的结果将可能是对切片的拷贝而非引用，这将使得对最初数组的赋值操作无法实现
        data_ori = data_ori.replace('-', np.nan)
        data_ori['Timestamp'] = pd.to_datetime(data_ori['Timestamp'], format="%Y.%m.%d_%H.%M.%S")
        # 去除重复日期，补全缺失日期
        data_ori = data_ori.drop_duplicates('Timestamp', keep='first')
        date_range = pd.date_range(start=data_ori['Timestamp'][0], end=data_ori['Timestamp'].iloc[-1], freq='S')
        data_ori = data_ori.set_index('Timestamp').reindex(index=date_range)
        # 插值补全LTE类型中的缺失数据
        data_ori['NetworkMode'] = data_ori['NetworkMode'].ffill()
        kinds_to_in = ['Speed', 'RSRP', 'RSRQ', 'RSSI', 'SNR', 'CQI', 'UL_bitrate', 'DL_bitrate']
        data_temp = data_ori[kinds_to_in].astype(np.float64)
        data_ori[kinds_to_in] = data_temp.interpolate('linear', axis=0, limit_direction='both')
        columns = cfg.features
        data_final = pd.DataFrame(columns=columns, dtype=np.float64)
        # 将网络类型用one-hot编码表示，并取窗口大小为5的滑动平均（对以dB或dBm为单位的数据项进行“指数平均”）
        for k in range(0, data_ori.shape[0] - cfg.collapse_interval + 1, cfg.collapse_interval):
            if data_ori.iloc[k]['NetworkMode'] == 'LTE':
                dt = [1, 0, 0]
            elif data_ori.iloc[k]['NetworkMode'] == 'EDGE':
                dt = [0, 0, 1]
            else:
                dt = [0, 1, 0]
            dt.append(np.mean(data_ori.iloc[k:k + cfg.collapse_interval]['Speed']))
            dt.append(expo_avg(data_ori.iloc[k:k + cfg.collapse_interval]['RSRP']))
            dt.append(expo_avg(data_ori.iloc[k:k + cfg.collapse_interval]['RSRQ']))
            dt.append(expo_avg(data_ori.iloc[k:k + cfg.collapse_interval]['RSSI']))
            dt.append(expo_avg(data_ori.iloc[k:k + cfg.collapse_interval]['SNR']))
            dt.append(np.mean(data_ori.iloc[k:k + cfg.collapse_interval]['CQI']))
            factor = {'kb': 1, 'kB': 8, 'Mb': 1024}
            dt.append(np.mean(data_ori.iloc[k:k + cfg.collapse_interval]['UL_bitrate']) / factor[cfg.unit])
            dt.append(np.mean(data_ori.iloc[k:k + cfg.collapse_interval]['DL_bitrate'])/factor[cfg.unit])
            data_final = pd.concat([data_final, pd.DataFrame(dt, index=columns).T], ignore_index=True)

        data_final.to_csv('./turned_dataset/LTE_Dataset/' + i + '/' + j)
        print('./dataset/' + i + '/' + j + '处理完毕，共' + str(data_final.shape[0]) + '项数据')
        # data.append(data_final)
print()
