''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MixedScore_MultiHeadAttention,MultiHeadAttentionMixedScore,EncodingBlock


__author__ = "Yu-Hsiang Huang"


class MixEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0, atten_mode = "softmax"):
        super(MixEncoderLayer, self).__init__()
        self.head_num = n_head

        self.Wq = nn.Linear(d_model, n_head * d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_head * d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_head * d_v, bias=False)
        self.mix_attn = MultiHeadAttentionMixedScore(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input,cost_mat=None ,k_v = None ):
        enc_input = enc_input
        # import pdb; pdb.set_trace()

        if k_v == None:
            enc_output,_ = self.mix_attn(enc_input,enc_input,enc_input,cost_mat)
        else:
            enc_output,_ = self.mix_attn(enc_input,k_v,k_v,cost_mat)

        enc_output = self.pos_ffn(enc_output)
        return enc_output
    

class Mix_flow_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0, atten_mode = "softmax"):
        super(Mix_flow_EncoderLayer, self).__init__()
        self.head_num = n_head

        self.mix_attn = EncodingBlock(n_head, d_model, d_k)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input,cost_mat=None ,k_v = None ):
        enc_input = enc_input
        # import pdb; pdb.set_trace()

        if k_v == None:
            enc_output= self.mix_attn(enc_input,enc_input,cost_mat)
        else:
            enc_output = self.mix_attn(enc_input,k_v,cost_mat)

        enc_output = self.pos_ffn(enc_output)
        return enc_output
    
class MixEncoderLayerWCross(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0, atten_mode = "softmax"):
        super(MixEncoderLayerWCross, self).__init__()
        self.head_num = n_head

        self.Wq = nn.Linear(d_model, n_head * d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_head * d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_head * d_v, bias=False)
        self.mix_attn = MultiHeadAttentionMixedScore(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.pos_ffn_1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_2 = PositionwiseFeedForward(d_model,d_inner,dropout=dropout)

    def forward(self, enc_input, cros_input ,cost_mat=None):
        enc_input = enc_input
        enc_output,_ = self.mix_attn(enc_input,enc_input,enc_input,cost_mat)
        enc_output = self.pos_ffn_1(enc_output)
        enc_output,_ = self.cross_attn(enc_output,cros_input,cros_input)
        enc_output = self.pos_ffn_2(enc_output)

        return enc_output
    
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0, atten_mode = "softmax"):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_input = enc_input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output

class CrossEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0, atten_mode = "softmax"):
        super(CrossEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input,kv, slf_attn_mask=None):
        enc_input = enc_input
        # import pdb; pdb.set_trace()
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, kv, kv, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0, atten_mode = "softmax"):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, atten_mode = atten_mode)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):

        dec_output, dec_enc_attn = self.enc_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_enc_attn


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed