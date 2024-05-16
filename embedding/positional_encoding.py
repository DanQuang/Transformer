import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, seq_len: int):
        super(PositionalEncoding, self).__init__()

        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # create a matrix shape [seq_len, embedding_dim]
        self.pe = torch.zeros((self.seq_len, self.embedding_dim))
        self.pe.requires_grad = False

        position = torch.arange(0, self.seq_len, dtype= torch.float).unsqueeze(1) # [seq_len, 1]
        div_term = torch.arange(0, self.embedding_dim, 2).float()

        self.pe[:, 0::2] = torch.sin(position / 10000.0 ** (div_term / self.embedding_dim))
        self.pe[:, 1::2] = torch.cos(position / 10000.0 ** (div_term / self.embedding_dim))

    def forward(self, x):
        # Shape x: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        return self.pe[:seq_len, :] # [seq_len, embedding_dim]