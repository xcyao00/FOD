import math
import torch
import torch.nn as nn
import numpy as np


class Attention2D(nn.Module):
    def __init__(self, 
                 seq_len=256,
                 d_model=512,
                 num_heads=8,
                 scale_factor=None,
                 dropout=0.0):
        super(Attention2D, self).__init__()
        
        self.seq_len = seq_len
        self.width = int(math.sqrt(seq_len))
        self.scale_factor = scale_factor or 1. / math.sqrt(d_model // num_heads)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.sigma_projection = nn.Linear(d_model, 2 * num_heads)
        self.out_projection = nn.Linear(d_model, d_model)
        
        distances_x = np.load('distances/distances_x_{}.npy'.format(seq_len))
        distances_y = np.load('distances/distances_y_{}.npy'.format(seq_len))
        distances_x = torch.from_numpy(distances_x)
        distances_y = torch.from_numpy(distances_y)
        self.distances_x = distances_x.cuda()
        self.distances_y = distances_y.cuda()

    def forward(self, query, key, value, return_attention=True):
        B, L, _ = query.shape
        _, S, _ = key.shape

        if return_attention:  # the sigma will be learned in the intra and inter correlation branches
            sigma = self.sigma_projection(query).view(B, L, self.num_heads, -1)
        query = self.query_projection(query).view(B, L, self.num_heads, -1)
        key = self.key_projection(key).view(B, S, self.num_heads, -1)
        value = self.value_projection(value).view(B, S, self.num_heads, -1)
        
        scores = torch.einsum("blhe,bshe->bhls", query, key)
    
        # attn: (N, n_heads, L, L)
        attn = self.scale_factor * scores

        if return_attention:
            sigma = sigma.transpose(1, 2)  # (B, L, n_heads, 2) ->  (B, n_heads, L, 2)
            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1  # can change these hyperparameter
            
            sigma1 = sigma[:, :, :, 0]  # (B, n_heads, L)
            sigma2 = sigma[:, :, :, 1]  # (B, n_heads, L)
            sigma1 = sigma1.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)  # (B, n_heads, L, L)
            sigma2 = sigma2.unsqueeze(-1).repeat(1, 1, 1, self.seq_len)  # (B, n_heads, L, L)
            
            # (B, n_heads, L, L)
            distances_x = self.distances_x.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
            distances_y = self.distances_y.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
            # gaussian distance prior
            target = 1.0 / (2 * math.pi * sigma1 * sigma2) * torch.exp(-distances_y / (2 * sigma1 ** 2) -distances_x / (2 * sigma2 ** 2))

        softmax_scores = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", softmax_scores, value)

        out = out.contiguous().view(B, L, -1)
        self.out_projection(out)

        if return_attention:
            return out, softmax_scores, target
        else:
            return out, None, None


