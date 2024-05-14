import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, padding_idx: int):
        super(InputEmbedding, self).__init__()

        self.embedding_dim = embedding_dim # d_model
        self.vocab_size = vocab_size 
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, self.padding_idx)

    def forward(self, x):
        # In paper, they multiply embedding weights to sqrt(embedding_dim)
        return self.embedding(x) * math.sqrt(self.embedding_dim)