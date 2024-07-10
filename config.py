# 数据预处理相关
RSRQ_fill = -2
SNR_fill = 0
CQI_fill = 0
RSSI_fill = -40
collapse_interval = 2  # 将原始数据中的多少个相邻时刻的数据合为一个
features = ['is4G', 'is3G', 'is2G', 'speed', 'RSRP', 'RSRQ', 'RSSI', 'SNR', 'CQI', 'UL_Bandwidth', 'bandwidth']
unit = 'kB'  # 带宽的单位，应在kB/kb/Mb中选择

# 数据集创建相关
train_ratio = 0.8
test_ratio = 0.15
seed = 1022
window_size = 150  # 采用滑动窗口法创建数据集时窗口的大小
window_step = 5
norm_type = 'std'
bins = [-1e-06, 0.03, 0.06, 0.12, 0.18, 0.24, 0.4]
batch_size = 32
num_state = len(bins)  # 状态数（=图中顶点数）
num_time_step = 7  # 划分的子序列数
adj_row_norm = True  # 邻接矩阵是否行归一化
load_graph = False  # 是否从文件中加载图而非计算
series_length = 40  # 实际输入模型的序列长度

# 模型输入部分相关
use_input_embedding = True
input_embedding_dim = 32
use_label_bins = False  # unknown
initial_state = 'embedding'  # choices=['range', 'mean_std', 'embedding', 'one_hot']

# GNN相关
message_passing = 'gcn'  # choices=['matrix', 'gcn']
message_passing_outdim = 32  # 特征向量维度
node_dim = 1  # 每个结点的向量的起始维度
gnn_layer = 'gcn'
gnn_layer_num = 3
multi_gnn = False  # 是只使用一个GNN然后各个子序列参数共享还是为多个子序列使用各自独立的GNN

# node attention相关
use_node_rnn = True  # 为True则使用RNN，否则使用时域Transformer
use_graph_rnn = False
graph_readout = 'none'
use_attention = True  # 是否使用注意力机制
use_layernorm = True  # 是否使用层归一化
atten_num_head = 8
atten_dropout = 0.1  # dropout中元素被设为0的概率
residual = True
d_qkv = 16  # 隐层向量的维度
evolve_layer = 'lstm'

# node rnn相关
node_rnn_layer_num = 2
node_rnn_hdim = 32
pool = 'none'
last_k_state = 1  # 根据最后几个时刻的状态来选取相应顶点的特征向量
use_all_node = True

# 序列RNN相关
series_model = 'transformer'  # 序列模型，可取lstm、gru、transformer、DenseNet
series_rnn_num_layer = 2
series_rnn_hdim = 64

# 序列Transformer相关
transformer_block_num = 5
transformer_token_dim = 32
transformer_num_tokens_per_variate = 3
transformer_out_dim = 64

use_parallel_encoder = True
fusion_type = 'weighted'  # 可选weighted、concat、state_only、series_only
linear_hdim = 32  # 融合输出时全连接层的隐层向量维度
state_attention_wdim = 100  # unknown
use_state_attention = False  # 是否使用“状态注意力”
use_multi_loss = False
topk_node = 8  # unknown

# 实验配置相关
train = True
continue_train = False
exp_type = 'StateGraph'  # 实验文件夹的路径
load_model_path = '2024-01-13_17-15-38_both(Tr+state)_weighted(3rd)'
weight_decay = 0.005
lr = 0.01
lr_scheduler = 'MultiStepLR'
miletones = [10, 20, 30, 40, 50, 60, 70, 80, 90]
gamma = 0.5
patience = 100
loss = 'MSE'
# loss_weight = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.64]
loss_weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
epoch = 100
model = 'StateGraph'
use_series = True
loss_weight1 = 0.2
loss_weight2 = 0.8  # 使用multi_loss时两个损失的加权权重

if __name__ == '__main__':
    import numpy as np

    pre = {}.fromkeys(['1', '2', '3'], np.array([1, 2, 3]))
    pre['1'] = np.array([1, 2])
    print()
    pre = {}.fromkeys(['1', '2', '3'], [])
    pre['1'].append(1)
    print()
    pre = {}.fromkeys(['1', '2', '3'], [])
    pre['1'] = [1, 2]
    print()
