import FFN
import LayerNormal
import mutiheadAttention
import torch
from torch import nn

class DecodeLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, drop_prob, n_head):
        super(DecodeLayer, self).__init__()

        self.attention = mutiheadAttention.multi_head_attention(d_model, n_head)
        self.norm1 = LayerNormal.LayerNormal(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.cross_attention = mutiheadAttention.multi_head_attention(d_model, n_head)
        self.norm2 = LayerNormal.LayerNormal(d_model)
        self.drop2 = nn.Dropout(drop_prob)

        self.ffn = FFN.PostionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormal.LayerNormal(d_model)
        self.drop3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention(dec, dec, dec, t_mask)
        x = self.drop1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(dec, enc, enc, s_mask)
            x = self.drop2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop3(x)
        x = self.norm3(x + _x)

        return x