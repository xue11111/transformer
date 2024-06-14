import decodeLayer
import TotalEmbedding
import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, n_head, ffn_hidden, n_layer, drop_prob, device):
        super(Decoder, self).__init__()

        self.embedding = TotalEmbedding.TotalEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [decodeLayer.DecodeLayer(d_model, ffn_hidden, drop_prob, n_head) for _ in range(n_layer)]
        )

        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)

        dec = self.fc(dec)
        return dec