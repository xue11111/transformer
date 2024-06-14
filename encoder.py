import encodeLayer
import TotalEmbedding
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, n_head, ffn_hidden, n_layer, drop_prob, device):
        super(Encoder, self).__init__()

        self.embedding = TotalEmbedding.TotalEmbedding(enc_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList(
            [encodeLayer.EncodeLayer(d_model, ffn_hidden, drop_prob, n_head) for _ in range(n_layer)]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)

        return x