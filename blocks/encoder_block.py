import torch
import torch.nn as nn

from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden, n_head, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(embedding_dim, hidden, dropout)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(x, x, x, src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. position-wise feed forward
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x

