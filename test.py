import torch
import encoder
import decoder
from transform import Transformer

# 定义编码器和解码器的参数
src_pad_idx = 0
trg_pad_idx = 0
enc_voc_size = 10000
dec_voc_size = 10000
max_len = 100
d_model = 512
n_head = 8
ffn_hidden = 2048
n_layers = 6
drop_prob = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建 Transformer 实例
model = Transformer(src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_len, d_model, n_head, ffn_hidden, n_layers, drop_prob, device).to(device)

# 创建示例输入张量
src = torch.randint(0, enc_voc_size, (2, 50)).to(device)  # 假设输入的形状为 (batch_size, src_len)
trg = torch.randint(0, dec_voc_size, (2, 50)).to(device)  # 假设目标的形状为 (batch_size, trg_len)

# 前向传播
output = model(src, trg)
print(output.shape)  # 输出形状应为 (batch_size, trg_len, dec_voc_size)
