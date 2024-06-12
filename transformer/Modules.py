import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, atten_mode = "softmax"):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.atten_mode = atten_mode

    def sinkhorn(self, tensors):
        tensors = F.sigmoid(tensors)

        for _ in range(5):
            row_sum = torch.sum(tensors, dim= 3).unsqueeze(3) + (1e-9)
            tensors = tensors / row_sum
            col_sum = torch.sum(tensors, dim= 2).unsqueeze(2) + (1e-9)
            tensors = tensors / col_sum

        return tensors

    def forward(self, q, k, v, cost_mat = None, mask=None):
        #Q K V [batchsize,num_head,len,dim]   the output is the same.
        if cost_mat is not None:
            cost_mat_score = cost_mat[:, None, :, :].expand(q.shape[0], q.shape[1], q.shape[2], q.shape[2])
        
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        if self.atten_mode == "softmax":
            attn = self.dropout(F.softmax(attn, dim=-1))
        else:
            attn = self.sinkhorn(attn)

        if mask is not None:
            attn =attn * mask

        # import pdb; pdb.set_trace()
        if cost_mat is not None:
            attn = attn*cost_mat_score
        
        output = torch.matmul(attn, v)

        return output, attn
