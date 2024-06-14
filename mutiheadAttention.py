import math
import torch
from torch import nn
import torch.functional as F

# batch time dimension
x = torch.randn(128, 64, 512)

d_model = 512
# 多头注意力机制
n_head = 8


class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(multi_head_attention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        # 多头注意力机制中的Q,K,V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v,mask=None):

        batch, time, dimension = q.shape
        # 划分多头
        n_d = self.d_model // self.n_head
        # 生成q,k,v
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 将q k v转换成batch,n_head,time,self.n_head
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        # 求出得分---q * k的转置除以n_d的平方根，再进行softmax处理后乘以v
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)

        #decoder中需要的mask
        if mask is not None:
            # mask = torch.tril(torch.ones(time, time, dtype=bool))
            score = score.masked_fill(mask == 0, float("-inf"))

        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        output = self.w_combine(score)

        return output

attention = multi_head_attention(d_model, n_head)
output = attention(x, x, x)
print(output,output.shape)






