import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from model.attention_layer import Attention


def feed_forward(dim, mult=4, dropout=0.):
    dim_inner = dim * mult
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )


class iTransformer(nn.Module):
    def __init__(self, lookback_len, num_variates, depth, dim, num_tokens_per_variate, out_dim, dim_head=32, heads=4,
                 attn_dropout=0.1, ff_dropout=0.1, ff_mult=3):
        super().__init__()
        self.lookback_len = lookback_len  # 时间序列的长度
        self.num_variates = num_variates  # 特征数
        self.depth = depth  # TrmBlock的数量
        self.dim = dim  # 将变量转成Token后，Token向量的维度
        self.out_dim = out_dim  # 输出向量的维度
        self.num_tokens_per_variate = num_tokens_per_variate  # 每个变量对应多少个Token
        # dim_head: 每个注意力头的q/k/v向量的维度
        # heads: 注意力的头数
        # attn_dropout: dropout的概率
        # ff_mult: 全连接层中的隐层向量维度是输入的多少倍

        self.mlp_in = nn.Sequential(
            nn.Linear(lookback_len, dim * num_tokens_per_variate),
            Rearrange('b v (n d) -> b (v n) d', n=num_tokens_per_variate),
            nn.LayerNorm(dim)
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_qkv=dim_head, n_heads=heads, attn_drop=attn_dropout),
                nn.LayerNorm(dim),
                feed_forward(dim, mult=ff_mult, dropout=ff_dropout),
                nn.LayerNorm(dim)
            ]))

        self.pred_head = nn.Sequential(
            Rearrange('b (v n) d -> b v (n d)', n=num_tokens_per_variate),
            nn.Linear(dim * num_tokens_per_variate, self.out_dim),
        )

    def forward(self, x):
        # x: (BS, Time, NumFeatures)/(b, t, v)
        x = torch.permute(x, (0, 2, 1))  # (b, v, t)

        # mlp to tokens
        x = self.mlp_in(x)

        # attention and feedforward layers
        for attn, attn_post_norm, ff, ff_post_norm in self.layers:
            x = attn(x) + x
            x = attn_post_norm(x)
            x = ff(x) + x
            x = ff_post_norm(x)

        pred_dict = self.pred_head(x)
        return pred_dict
