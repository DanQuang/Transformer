import torch.nn as nn

from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden, n_head, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm1 = LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attention = MultiHeadAttention(embedding_dim, n_head)
        self.norm2 = LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(embedding_dim, hidden, dropout)
        self.norm3 = LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, enc, dec, src_mask, trg_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q= dec, k= dec, v= dec, mask= trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. compute encoder and decoder attention
        if enc is not None:
            _x = x
            x = self.cross_attention(q= x, k= enc, v= enc, mask= src_mask)

            # add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 4. feed forward
        _x = x
        x = self.ffn(x)

        # 5. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x