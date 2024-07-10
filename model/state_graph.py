import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.attention_layer import NodeAttentionLayer
from model.iTransformer import iTransformer
from model.densenet import DenseNet
import config as cfg
import copy
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def attention_select(bw_bins, last_one):
    bw_bins = copy.copy(bw_bins.reshape(-1))
    bw_bins[0] = -np.inf
    bw_bins = np.concatenate([bw_bins, np.array([np.inf])], axis=0)
    state = pd.cut(pd.Series(last_one.cpu()), bw_bins, labels=False).values
    return state


class StateGraph(nn.Module):
    def __init__(self, device):
        super(StateGraph, self).__init__()
        self.device = device
        self.num_state = cfg.num_state
        self.num_step = cfg.num_time_step
        self.node_dim = cfg.node_dim
        self.gcn_outdim = cfg.message_passing_outdim
        self.linear_hdim = cfg.linear_hdim

        # input embedding
        self.input_embedding_dim = cfg.input_embedding_dim
        self.node_rnn_hdim = self.input_embedding_dim
        self.gnn_input_size = self.input_embedding_dim
        self.input_embedding = nn.Embedding(cfg.num_state, self.input_embedding_dim)

        # GNN
        if cfg.multi_gnn:
            self.message_passing_layer1 = nn.ModuleList(
                [nn.Linear(self.gnn_input_size, self.gcn_outdim, bias=False) for i in range(self.num_step)])
        else:
            self.message_passing_layer1 = nn.ModuleList(
                [nn.Linear(self.gnn_input_size, self.gcn_outdim, bias=False) for i in range(cfg.gnn_layer_num)])
            self.message_passing_layer2 = nn.ModuleList(
                [nn.Linear(self.gnn_input_size, self.gcn_outdim, bias=True) for i in range(cfg.gnn_layer_num)])

        # temporal layer
        self.linear_input_dim = 0
        if cfg.evolve_layer in ['lstm', 'gru']:
            if cfg.evolve_layer == 'gru':
                self.linear_input_dim += self.node_rnn_hdim
                self.node_rnn = nn.GRU(self.gcn_outdim, self.node_rnn_hdim, cfg.node_rnn_layer_num)
            elif cfg.evolve_layer == 'lstm':
                self.linear_input_dim += self.node_rnn_hdim
                self.node_rnn = nn.LSTM(self.gcn_outdim, self.node_rnn_hdim, cfg.node_rnn_layer_num)

        if cfg.use_node_rnn:
            self.node_rnn_fc = nn.Linear(self.node_rnn_hdim * cfg.num_state, self.linear_input_dim)
        else:
            self.state_transformer = iTransformer(self.num_state * self.input_embedding_dim, self.num_step,
                                                  cfg.transformer_block_num,
                                                  cfg.transformer_token_dim,
                                                  cfg.transformer_num_tokens_per_variate,
                                                  cfg.transformer_out_dim)  # self.linear_input_dim
            self.node_rnn_fc = nn.Linear(cfg.transformer_out_dim, self.linear_input_dim)
        # parallel rnn
        if cfg.use_parallel_encoder:
            self.series_rnn_hdim = cfg.series_rnn_hdim
            if cfg.series_model == 'gru':
                self.series_rnn = nn.GRU(len(cfg.features), self.series_rnn_hdim, cfg.series_rnn_num_layer,
                                         batch_first=True)
            elif cfg.series_model == 'lstm':
                self.series_rnn = nn.LSTM(len(cfg.features), self.series_rnn_hdim, cfg.series_rnn_num_layer,
                                          batch_first=True)
            elif cfg.series_model == 'transformer':
                self.series_transformer = iTransformer(cfg.series_length, len(cfg.features), cfg.transformer_block_num,
                                                       cfg.transformer_token_dim,
                                                       cfg.transformer_num_tokens_per_variate,
                                                       cfg.transformer_out_dim)
            elif cfg.series_model == 'DenseNet':
                self.series_DenseNet = DenseNet(cfg.series_length, cfg.transformer_out_dim)
            else:
                raise RuntimeError('未知的序列模型。')

            if cfg.fusion_type == 'concat':
                self.fc1 = nn.Linear(cfg.transformer_out_dim, self.linear_hdim)
                # self.linear_input_dim + self.series_rnn_hdim
                self.fc2 = nn.Linear(self.linear_hdim, 1)
                self.fc3 = nn.Linear(17, 10)
            elif cfg.fusion_type == 'weighted':
                self.w1 = nn.Linear(self.linear_input_dim, self.linear_hdim, bias=False)  # (bs, linear_input_dim)
                if cfg.series_model == 'transformer':
                    self.w2 = nn.Linear(cfg.transformer_out_dim, self.linear_hdim, bias=False)  # (bs, num_vari, outdim)
                    self.fc = nn.Linear(self.linear_hdim, 1, bias=False)
                else:
                    self.w2 = nn.Linear(self.series_rnn_hdim, self.linear_hdim, bias=False)
                    self.fc = nn.Linear(self.linear_hdim, len(cfg.features), bias=False)
            elif cfg.fusion_type == 'state_only':
                self.fc = nn.Linear(self.linear_input_dim, len(cfg.features), bias=False)
            elif cfg.fusion_type == 'series_only':
                if cfg.series_model == 'transformer':
                    self.fc1 = nn.Linear(cfg.transformer_out_dim, self.linear_hdim, bias=False)
                    self.fc2 = nn.Linear(self.linear_hdim, 1, bias=False)
                elif cfg.series_model == 'DenseNet':
                    self.fc1 = nn.Linear(cfg.transformer_out_dim, self.linear_hdim, bias=False)
                    self.fc2 = nn.Linear(self.linear_hdim, 1, bias=False)
                else:
                    self.fc1 = nn.Linear(self.series_rnn_hdim, self.linear_hdim, bias=False)
                    self.fc2 = nn.Linear(self.linear_hdim, len(cfg.features), bias=False)
            else:
                raise RuntimeError('未知的融合类型。')
        else:
            self.fc1 = nn.Linear(self.linear_input_dim * cfg.last_k_state, self.linear_hdim)
            self.fc2 = nn.Linear(self.linear_hdim, 1)

        if cfg.use_label_bins:
            self.fc2 = nn.Linear(self.linear_hdim, self.num_state)

        if cfg.use_attention:
            self.node_attention = NodeAttentionLayer(self.num_state, self.gcn_outdim, cfg.atten_num_head, cfg.d_qkv,
                                                     cfg.atten_dropout, cfg.residual)

        if cfg.use_layernorm:
            self.node_rnn_layernorm = nn.LayerNorm(self.node_rnn_hdim, eps=1e-6)
            if cfg.use_parallel_encoder:
                self.series_rnn_layernorm = nn.LayerNorm(self.series_rnn_hdim, eps=1e-6)

    def local_structure_block(self, weight, node_feature, t=None):
        weight = weight.squeeze().view(-1, cfg.num_state, cfg.num_state)
        if cfg.multi_gnn:
            h = self.message_passing_layer1[t](node_feature)
            h = torch.bmm(weight, h)
            return h
        # else:
        #     h = self.message_passing_layer1(node_feature)
        #     h = torch.bmm(weight, h)
        #     return h

    def attention_block(self, node_structure_outputs):
        node_structure_outputs = self.node_attention(node_structure_outputs)
        return node_structure_outputs

    def forward(self, graphs, series=None, bins=None):
        node_structure_outputs = []
        bw_bins = bins
        weights = graphs  # weights: (num_series, bs, num_state^2, 1)
        batch_size = weights[0].size(0)

        original_node_feat = torch.repeat_interleave(torch.arange(cfg.num_state).reshape(-1, 1).unsqueeze(0),
                                                     batch_size, dim=0)
        original_node_feat = original_node_feat.to(device)
        node_feat = self.input_embedding(original_node_feat).squeeze()  # (bs, num_state, embedding_dim)

        # 结构信息（GNN）
        for t in range(self.num_step):
            if cfg.multi_gnn:  # 多个单层
                x = node_feat.reshape(batch_size * self.num_state, -1)
                node_structure_output = self.local_structure_block(weights[t].to(device), node_feat, t)
                node_structure_output = node_structure_output.reshape(batch_size * self.num_state, -1)  # (B*N, D)
                node_structure_output = F.relu(node_structure_output)
                node_structure_output = torch.add(x, node_structure_output)
                node_structure_outputs.append(node_structure_output[None, :, :])
            else:  # 一个多层
                weight = weights[t].to(device).squeeze().view(-1, cfg.num_state, cfg.num_state)
                for i in range(cfg.gnn_layer_num):
                    x = node_feat
                    h1 = self.message_passing_layer1[i](node_feat)
                    h1 = torch.bmm(weight, h1)
                    h2 = self.message_passing_layer2[i](node_feat)
                    node_feat = F.relu(h1 + h2) + x
                node_structure_outputs.append(node_feat.reshape(batch_size * self.num_state, -1)[None, :, :])
        node_structure_outputs = torch.concat(node_structure_outputs, dim=0)  # (num_time_step, bs*state, D)

        # 节点之间的spatial attention
        if cfg.use_attention:
            # (num_time_step, bs*state, D)
            node_structure_outputs = node_structure_outputs + self.attention_block(node_structure_outputs)

        # node rnn
        if cfg.use_node_rnn:
            # node_structure_outputs = node_structure_outputs.reshape(self.num_step, batch_size, -1)
            node_rnn_output, _ = self.node_rnn(node_structure_outputs)
            node_rnn_output = node_rnn_output[-1, :, :]  # (bs*num_state, embedding_len)
            if cfg.use_layernorm:
                node_rnn_output = self.node_rnn_layernorm(node_rnn_output)
            if cfg.pool == 'none':
                node_rnn_output = node_rnn_output
        else:
            node_structure_outputs = node_structure_outputs.reshape(self.num_step, batch_size, self.num_state, -1)
            node_structure_outputs = node_structure_outputs.reshape(self.num_step, batch_size, -1).permute(1, 2, 0)
            node_rnn_output = self.state_transformer(node_structure_outputs)[:, -1, :]
            # node_rnn_output = self.state_transformer(node_structure_outputs)
            # (bs, out_dim)

        full_series = series  # (bs, series_length, num_features)
        series = series[:, :, -1]

        if cfg.use_node_rnn:
            if not cfg.use_all_node:
                # 选择最后一个时刻对应的状态
                state_atten_output = []
                for i in range(1, 1 + cfg.last_k_state):
                    last_state = series[:, -i]
                    idx = attention_select(bw_bins, last_state)
                    idx += np.arange(0, self.num_state * series.shape[0], self.num_state)
                    state_atten_output.append(node_rnn_output[idx, :])
                state_atten_output = torch.concat(state_atten_output, dim=1)
            else:
                node_rnn_output = node_rnn_output.reshape(batch_size, self.num_state, -1)
                node_rnn_output = node_rnn_output.reshape(batch_size, -1)
                state_atten_output = self.node_rnn_fc(node_rnn_output)
        else:
            state_atten_output = self.node_rnn_fc(node_rnn_output)
            # state_atten_output = node_rnn_output
        if cfg.use_parallel_encoder:
            if cfg.series_model in ['lstm', 'gru']:
                series_rnn_output, _ = self.series_rnn(full_series)
                series_rnn_output = series_rnn_output[:, -1, :]
            elif cfg.series_model == 'DenseNet':
                series_rnn_output = self.series_DenseNet(series)
            else:
                series_rnn_output = self.series_transformer(full_series)
            # if cfg.use_layernorm:
            #     series_rnn_output = self.series_rnn_layernorm(series_rnn_output)
            if cfg.fusion_type == 'weighted':
                state_atten_output = state_atten_output.unsqueeze(1).repeat(1, len(cfg.features), 1)
                prediction = self.w1(state_atten_output) + self.w2(series_rnn_output)
                # prediction = self.w1(state_atten_output)
                prediction = F.relu(prediction)
                prediction = self.fc(prediction).squeeze()
            elif cfg.fusion_type == 'concat':
                # state_atten_output = state_atten_output.unsqueeze(1).repeat(1, len(cfg.features), 1)
                prediction = torch.concat([series_rnn_output, state_atten_output], dim=1)
                prediction = self.fc1(prediction)
                prediction = F.relu(prediction)
                prediction = self.fc3(self.fc2(prediction).squeeze())
            elif cfg.fusion_type == 'state_only':
                prediction = self.fc(state_atten_output).squeeze()
            elif cfg.fusion_type == 'series_only':
                prediction = self.fc1(series_rnn_output)
                prediction = F.relu(prediction)
                prediction = self.fc2(prediction).squeeze()
            else:
                raise RuntimeError('未知的融合类型。')
        else:
            prediction = self.decoder(state_atten_output)
        return prediction
