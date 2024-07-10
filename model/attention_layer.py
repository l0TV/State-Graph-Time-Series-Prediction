import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeAttentionLayer(nn.Module):
    def __init__(self,
                 num_state,
                 input_dim,
                 n_heads,
                 # num_time_steps,
                 dim_qkv,
                 attn_drop,
                 residual):
        super(NodeAttentionLayer, self).__init__()
        self.num_state = num_state
        self.n_heads = n_heads
        # self.num_time_steps = num_time_steps
        self.residual = residual
        self.d_qkv = dim_qkv

        # define weights
        # self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, n_heads * dim_qkv))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, n_heads * dim_qkv))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, n_heads * dim_qkv))
        # ff
        # self.lin = nn.Linear(input_dim, input_dim, bias=True)
        self.lin = nn.Linear(n_heads * dim_qkv, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.xavier_init()

    def forward(self, inputs):
        """In:  node_structure_output (of StructuralModule at each snapshot):= [T, B*node_num, D]"""
        # 1: Add position embeddings to input
        """  不用位置编码
        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F]
        """
        residual = inputs

        # 2: Query, Key based multi-head self attention.
        # torch.tensordot等价于torch.matmul
        q = torch.tensordot(inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F] [T, B*node_num, head*d_qkv]
        k = torch.tensordot(inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F] [T, B*node_num, head*d_qkv]
        v = torch.tensordot(inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F] [T, B*node_num, head*d_qkv]

        '''
        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        '''

        num_timestep = inputs.size(0)
        batch_size = int(inputs.size(1) / self.num_state)
        num_state = self.num_state
        q_ = q.view(num_timestep, batch_size, num_state, self.n_heads, self.d_qkv)  # [T, B, node_num, nhead, d_qkv]
        k_ = k.view(num_timestep, batch_size, num_state, self.n_heads, self.d_qkv)  # [T, B, node_num, nhead, d_qkv]
        v_ = v.view(num_timestep, batch_size, num_state, self.n_heads, self.d_qkv)  # [T, B, node_num, nhead, d_qkv]
        q_, k_, v_ = q_.transpose(2, 3), k_.transpose(2, 3), v_.transpose(2, 3)  # [T, B, nhead, node_num, d_qkv]

        # todo
        # outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = torch.matmul(q_, k_.permute(0, 1, 2, 4, 3))  # [T, B, head, node_num, node_num]
        # outputs = outputs / (self.num_time_steps ** 0.5)
        outputs = outputs / (self.d_qkv ** 0.5)  # [T, B, head, node_num, node_num]

        '''
        没有mask，这是位置无关的attention
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        '''

        # outputs = F.softmax(outputs, dim=2)
        outputs = F.softmax(outputs, dim=-1)  # [T, B, head, node_num, node_num]
        self.attn_wts_all = outputs  # [h*N, T, T] [T, B, head, node_num, node_num]

        outputs = self.attn_dp(outputs)
        '''
        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        '''
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h] [T, B, head, node_num, d_qkv]
        # outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
        #                     dim=2)  # [N, T, F]
        outputs = outputs.transpose(2, 3).contiguous().view(num_timestep, batch_size, num_state, -1)

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        outputs = outputs.view(num_timestep, batch_size * num_state, -1)
        if self.residual:
            outputs = outputs + residual
        outputs = self.layer_norm(outputs)  # todo layer_norm不确定要不要
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        # return outputs + inputs
        return outputs

    def xavier_init(self):
        # nn.init.xavier_uniform_(self.position_embeddings)
        # nn.init.xavier_uniform_(self.Q_embedding_weights)
        # nn.init.xavier_uniform_(self.K_embedding_weights)
        # nn.init.xavier_uniform_(self.V_embedding_weights)
        nn.init.kaiming_uniform_(self.Q_embedding_weights)
        nn.init.kaiming_uniform_(self.K_embedding_weights)
        nn.init.kaiming_uniform_(self.V_embedding_weights)
        # nn.init.kaiming_uniform_()


class Attention(nn.Module):
    def __init__(self, input_dim, dim_qkv, n_heads, attn_drop):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.d_qkv = dim_qkv

        # define weights
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, n_heads * dim_qkv))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, n_heads * dim_qkv))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, n_heads * dim_qkv))

        # dropout and init
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

        self.feed_forward = nn.Sequential(
            nn.Linear(n_heads * dim_qkv, input_dim),
            nn.Dropout(attn_drop),
        )

    def xavier_init(self):
        nn.init.kaiming_uniform_(self.Q_embedding_weights)
        nn.init.kaiming_uniform_(self.K_embedding_weights)
        nn.init.kaiming_uniform_(self.V_embedding_weights)

    def forward(self, inputs):
        # 1: Query, Key based multi-head self attention.
        # [BS, F, input_dim]*[input_dim, head*d_qkv] => [BS, F, head*d_qkv]
        q = torch.tensordot(inputs, self.Q_embedding_weights, dims=([2], [0]))
        k = torch.tensordot(inputs, self.K_embedding_weights, dims=([2], [0]))
        v = torch.tensordot(inputs, self.V_embedding_weights, dims=([2], [0]))

        batch_size = inputs.size(0)
        num_features = inputs.size(1)
        q_ = q.reshape(batch_size, num_features, self.n_heads, self.d_qkv)
        k_ = k.reshape(batch_size, num_features, self.n_heads, self.d_qkv)
        v_ = v.reshape(batch_size, num_features, self.n_heads, self.d_qkv)
        q_, k_, v_ = q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2)  # [BS, n_head, feature_num, d_qkv]

        outputs = torch.matmul(q_, k_.permute(0, 1, 3, 2))  # [BS, n_head, feature_num, feature_num]
        outputs = outputs / (self.d_qkv ** 0.5)

        outputs = F.softmax(outputs, dim=-1)

        outputs = self.attn_dp(outputs)

        outputs = torch.matmul(outputs, v_)  # [BS, n_head, feature_num, d_qkv]
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, num_features, -1)

        # 6: Feedforward
        outputs = self.feed_forward(outputs)
        return outputs
