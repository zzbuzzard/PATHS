import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

from data_utils.patch_batch import PatchBatch
import utils


class ABMIL(nn.Module):
    def __init__(self, config, train_config):
        super().__init__()
        self.dim = config.patch_embed_dim
        num_logits = train_config.num_logits()
        self.gate1 = torch.nn.Sequential(torch.nn.Linear(self.dim, 1, bias=False), torch.nn.Tanh()).eval()
        self.gate2 = torch.nn.Sequential(torch.nn.Linear(self.dim, 1, bias=False), torch.nn.Sigmoid()).eval()
        self.final_project = torch.nn.Linear(self.dim, num_logits, bias=False)

    def forward(self, data: PatchBatch):
        xs = data.fts                               # B x N x D
        a1 = utils.apply_to_non_padded(self.gate1, xs, data.valid_inds, 1)  # B x N x 1
        a2 = utils.apply_to_non_padded(self.gate2, xs, data.valid_inds, 1)  # B x N x 1
        a = a1 * a2  # B x N x 1

        # Ensure post-softmax importance of 0 at padded locations
        a[~data.valid_inds] = -10000

        a = torch.softmax(a[..., 0], dim=-1)        # B x N
        z = torch.sum(a[..., None] * xs, dim=1)     # B x N x D -> B x D
        res = self.final_project(z)

        return res


# Copied from TransMIL codebase
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


# Copied from TransMIL codebase
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


# Adapted from TransMIL codebase
class TransMIL(nn.Module):
    def __init__(self, config, train_config):
        super(TransMIL, self).__init__()
        dim = config.transformer_dim
        self.pos_layer = PPEG(dim=dim)
        self._fc1 = nn.Sequential(nn.Linear(config.patch_embed_dim, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.n_classes = train_config.num_logits()
        self.layer1 = TransLayer(dim=dim)
        self.layer2 = TransLayer(dim=dim)
        self.norm = nn.LayerNorm(dim)
        self._fc2 = nn.Linear(dim, self.n_classes)

    def forward(self, data: PatchBatch):
        h = data.fts.float()  # B x N x D

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]

        return logits
