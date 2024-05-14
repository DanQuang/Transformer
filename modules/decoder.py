import torch.nn as nn

from blocks.decoder_block import DecoderBlock
from embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, dec_pad_idx, seq_len, embedding_dim, hidden, n_head, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size= dec_vocab_size,
                                              embedding_dim= embedding_dim,
                                              seq_len= seq_len,
                                              dropout= dropout,
                                              padding_idx= dec_pad_idx)
        
        self.layers = nn.ModuleList([DecoderBlock(embedding_dim= embedding_dim,
                                                  hidden= hidden,
                                                  n_head= n_head,
                                                  dropout= dropout)
                                    for _ in range(n_layers)])
        

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.embedding(trg)

        for layer in self.layers:
            trg = layer(enc_src, trg, src_mask, trg_mask)

        return trg
