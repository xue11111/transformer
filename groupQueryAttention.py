import torch
import torch.nn as nn
import math


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model,n_heads, n_groups):
        super(GroupQueryAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups


        assert d_model % n_heads == 0
        self.n_heads_group = self.n_heads // self.n_groups
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_v = nn.Linear(d_model, self.n_groups * self.head_dim)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def expand(self,data):
        batch, time = data.shape[0],  data.shape[2]
        data = data[:,:,None,:,:].expand(batch,self.n_groups,self.n_heads_group,time,self.head_dim).contiguous()
        data = data.view(batch,self.n_groups * self.n_heads_group, time, self.head_dim)

    def forward(self,q,k,v,mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch = q.shape[0]
        q = q.view(batch,-1,self.n_groups * self.n_heads_group,self.head_dim).permute(0,2,1,3)
        k = k.view(batch,-1,self.n_groups,self.head_dim).permute(0,2,1,3)
        v = v.view(batch,-1,self.n_groups,self.head_dim).permute(0,2,1,3)
        k = self.expand(k)
        v = self.expand(v)
        score = q @ k.transpose(2,3) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0,-1e9)
        score = self.softmax(score) @ v
        score = score.permute(0,2,1,3).view(batch,-1,self.d_model)
        output = self.w_combine(score)
        return  output
