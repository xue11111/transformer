import FFN
import LayerNormal
import mutiheadAttention
import torch
from torch import nn

class EncodeLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, drop_prob, n_head):
        super(EncodeLayer, self).__init__()

        self.attention = mutiheadAttention.multi_head_attention(d_model, n_head)
        self.norm1 = LayerNormal.LayerNormal(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = FFN.PostionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormal.LayerNormal(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.drop1(x)

        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.drop2(x)

        x = self.norm2(_x + x)
        return x