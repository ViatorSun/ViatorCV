#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2021.4
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :






''' Define the Layers '''
import torch.nn as nn
import torch
from   SubLayers import MultiHeadAttention, PositionwiseFeedForward




# 实现图26中的一个EncoderLayer，具体的结构如图19所示
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn  = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn( enc_input, enc_input, enc_input, mask=slf_attn_mask )
        enc_output               = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn



# 实现图26中的一个DecoderLayer，具体的结构如图19所示
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn  = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward( self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn( dec_input, dec_input, dec_input, mask=slf_attn_mask )
        dec_output, dec_enc_attn = self.enc_attn( dec_output, enc_output, enc_output, mask=dec_enc_attn_mask )
        dec_output               = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


