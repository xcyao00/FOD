import os
import pickle
import math
import torch
import torch.nn as nn

from .attention import Attention2D
from .embedding import Embedding2D


class Mlp(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, 
                 self_attention,
                 cross_attention,
                 d_model,
                 d_feed_foward=None,
                 dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        d_feed_foward = d_feed_foward or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ffn = Mlp(d_model, d_feed_foward, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #self.ffn_proj = Mlp(2 * d_model, d_feed_foward, out_features=d_model, drop=dropout)

    def forward(self, x, ref_x=None, with_intra=True, with_inter=True):
        if with_intra and with_inter:  # intra correlation + inter correlation
            new_x, intra_corr, intra_target = self.self_attention(
                x, x, x, 
                return_attention=True
            )
            new_x = x + self.dropout(new_x)
        
            new_ref_x, inter_corr, inter_target = self.cross_attention(
                x, ref_x, ref_x,
                return_attention=True
            )
            ref_x = x + self.dropout(new_ref_x) 
            new_x = new_x - ref_x  # I2Correlation: residual input
            # or concatenation input
            # new_x = torch.cat([new_x, ref_x], dim=-1)
            # new_x = self.ffn_proj(new_x)
        elif with_inter:  # only inter correlation
            new_x, inter_corr, inter_target = self.cross_attention(
                x, ref_x, ref_x,
                return_attention=True
            )
            new_x = x + self.dropout(new_x) 
        elif with_intra:  # only intra correlation
            new_x, intra_corr, intra_target = self.self_attention(
                x, x, x, 
                return_attention=True
            )
            new_x = x + self.dropout(new_x)
        else:  # only patch-wise reconstruction
            new_x, _, _ = self.self_attention(
                x, x, x, 
                return_attention=False
            )
            new_x = x + self.dropout(new_x)
              
        y = x = self.norm1(new_x)
        y = self.ffn(y)
        out = self.norm2(x + y)

        if with_intra and with_inter:
            return out, intra_corr, intra_target, inter_corr, inter_target
        elif with_intra:
            return out, intra_corr, intra_target, None, None
        elif with_inter:
            return out, None, None, inter_corr, inter_target
        else:
            return out, None, None, None, None


class Encoder(nn.Module):
    def __init__(self, encode_layers, norm_layer=None):
        super(Encoder, self).__init__()
        
        self.encode_layers = nn.ModuleList(encode_layers)
        self.norm = norm_layer

    def forward(self, x, ref_x=None, with_intra=True, with_inter=True):
        intra_corrs_list = []
        intra_targets_list = []
        inter_corrs_list = []
        inter_targets_list = []
        for layer in self.encode_layers:
            x, intra_corrs, intra_targets, inter_corrs, inter_targets = layer(x, ref_x, with_intra, with_inter)
            intra_corrs_list.append(intra_corrs)
            intra_targets_list.append(intra_targets)
            inter_corrs_list.append(inter_corrs)
            inter_targets_list.append(inter_targets)

        if self.norm is not None:
            x = self.norm(x)

        return x, intra_corrs_list, intra_targets_list, inter_corrs_list, inter_targets_list


class FOD(nn.Module):
    def __init__(self,
                 seq_len,
                 in_channels,
                 out_channels,
                 d_model=512,
                 n_heads=4,
                 n_layers=3,
                 d_feed_foward_scale=4,
                 dropout=0.0, 
                 args=None):
        super(FOD, self).__init__()
        
        d_feed_foward = d_model * d_feed_foward_scale

        # embedding
        h = w = int(math.sqrt(seq_len))
        self.embedding = Embedding2D(in_channels, d_model, dropout, h=h, w=w)

        self.with_intra = args.with_intra
        self.with_inter = args.with_inter
        if self.with_inter:
            # changing here for non 256x256 input
            mappings = {4096: 'layer0', 1024: 'layer1', 256: 'layer2', 64: 'layer3'}
            layer_name = mappings[seq_len]
        
            self.ref_embedding = Embedding2D(in_channels, d_model, dropout, h=h, w=w, with_pos_embed=True, device=args.device)
            ref_feature_filepath = os.path.join(args.rfeatures_path, '%s.pkl' % args.class_name)
            with open(ref_feature_filepath, 'rb') as f:
                ref_feats = pickle.load(f)
            ref_feats = ref_feats[layer_name]
            self.ref_feats = torch.from_numpy(ref_feats[0] if isinstance(ref_feats, list) else ref_feats).to(args.device)
            self.ref_feats = self.ref_feats.unsqueeze(0).repeat([args.batch_size, 1, 1]).permute(0, 2, 1)  # (N, L, dim)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Attention2D(
                        seq_len, d_model, n_heads, dropout=dropout, device=args.device),
                    Attention2D(
                        seq_len, d_model, n_heads, dropout=dropout, device=args.device),
                    d_model,
                    d_feed_foward,
                    dropout=dropout
                ) for l in range(n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, out_channels, bias=True)

    def forward(self, x, train=True):
        # (N, L, dim) -> (N, L, d_model)
        emb = self.embedding(x)
        if self.with_inter:
            ref_emb = self.ref_embedding(self.ref_feats if train else self.ref_feats[0:1, :, :])
            emb, intra_corrs, intra_targets, inter_corrs, inter_targets = self.encoder(emb, ref_emb, self.with_intra, self.with_inter)
        else:
            emb, intra_corrs, intra_targets, inter_corrs, inter_targets = self.encoder(emb, None, self.with_intra, self.with_inter)
        # (N, L, d_model) -> (N, L, dim)
        x_out = self.projection(emb)

        return x_out, intra_corrs, intra_targets, inter_corrs, inter_targets
        
