import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, h=16, w=16, device=torch.device('cuda')):
        super(PositionalEmbedding, self).__init__()

        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
        self.d_model = d_model
        self.h = h
        self.w = w

        pos_embed = torch.zeros(self.d_model, self.h, self.w)
        # Each dimension use half of D
        half_d_model = self.d_model // 2
        div_term = torch.exp(torch.arange(0.0, half_d_model, 2) * -(math.log(1e4) / half_d_model))
        pos_w = torch.arange(0.0, self.w).unsqueeze(1)
        pos_h = torch.arange(0.0, self.h).unsqueeze(1)
        pos_embed[0:half_d_model:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.h, 1)
        pos_embed[1:half_d_model:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.h, 1)
        pos_embed[half_d_model::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.w)
        pos_embed[half_d_model+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.w)

        self.pos_embed = pos_embed.to(device)

    def forward(self):
        return self.pos_embed


class ProjectEmbedding(nn.Module):
    def __init__(self, in_channels, d_model):
        super(ProjectEmbedding, self).__init__()
        
        self.project = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.project(x.permute(0, 2, 1)).transpose(1, 2)
        
        return x


class Embedding2D(nn.Module):
    def __init__(self, in_channels, d_model, dropout=0.0, h=16, w=16, with_pos_embed=True, device=torch.device('cuda')):
        super(Embedding2D, self).__init__()

        self.project_embedding = ProjectEmbedding(in_channels, d_model)
        if with_pos_embed:
            self.position_embedding = PositionalEmbedding(d_model, h=h, w=w, device=device)
        self.with_pos_embed = with_pos_embed

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.project_embedding(x)

        if self.with_pos_embed:
            pos_embed = self.position_embedding()
            pos_embed = pos_embed.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[-1])

            x = x + pos_embed
        
        return self.dropout(x)


