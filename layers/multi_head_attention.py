import torch
import torch.nn as nn
from scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        assert embedding_dim % n_head == 0, "embedding_dim is not divisible by n_head"

        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_concat = nn.Linear(embedding_dim, embedding_dim)

        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask= None):
        # 1. dot product
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # [batch_size, seq_len, embedding_dim]
        
        # 2. split tensor by numbers of head
        q, k, v = self.split(q), self.split(k), self.split(v) # [batch_size, head, seq_len, d_tensor]

        # 3. scale dot product to compute similarity
        out, score = self.attention(q, k ,v, mask= mask)

        # 4. concat an pass a linear layer
        batch_size, head, seq_len, d_tensor = out.size()
        embedding_dim = head * d_tensor
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        out = self.w_concat(out)

        return out
    
    def split(self, tensor: torch.Tensor):
        """
            split tensor by number of head

            :param tensor: [batch_size, seq_len, embedding_dim]
            :return: [batch_size, head, seq_len, d_tensor]
        """
        batch_size, seq_len, embedding_dim = tensor.size()
        d_head = embedding_dim // self.n_head
        tensor = tensor.view(batch_size, seq_len, self.n_head, d_head).transpose(1, 2)

        return tensor