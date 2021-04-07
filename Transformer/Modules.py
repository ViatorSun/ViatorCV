#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2021.4
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :




import torch
import torch.nn as nn
import torch.nn.functional as F



# 实现的是图22的操作
# 先令 QK^t ，再对结果按位乘以 Mask矩阵，再做 Softmax操作，最后的结果与 V相乘，得到self-attention的输出
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout     = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn   = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn   = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
