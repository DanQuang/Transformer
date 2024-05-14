import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding
from token_embedding import InputEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_len, dropout, padding_idx):
        super(TransformerEmbedding, self).__init__()

        self.tok_emb = InputEmbedding(embedding_dim, vocab_size, padding_idx)
        self.pos_emb = PositionalEncoding(embedding_dim, seq_len)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        return self.dropout(tok_emb + pos_emb)