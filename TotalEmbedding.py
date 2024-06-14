import postionEmbedding
import tokenEmbedding
import torch
from torch import nn


class TotalEmbedding(nn.Module):
    def __init__(self,vocab_size, d_model, maxlen, drop_prob, device):
        super(TotalEmbedding, self).__init__()

        self.pos_emb = postionEmbedding.PostionEmbedding(d_model, maxlen, device)
        self.tok_emb = tokenEmbedding.TokenEmbedding(vocab_size,d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)