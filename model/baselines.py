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



# Adapted from ILRA codebase
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K):

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, _ = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat


class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret = self.mha(global_embedding, X)
        return ret


class ILRA(nn.Module):
    def __init__(self, config, train_config):
        super().__init__()
        # stack multiple GAB block
        gab_blocks = []
        for idx in range(config.num_layers):
            block = GAB(dim_in=config.patch_embed_dim if idx == 0 else config.hidden_feat,
                        dim_out=config.hidden_feat,
                        num_heads=config.num_heads,
                        num_inds=config.topk,
                        ln=config.ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=config.hidden_feat, num_heads=config.num_heads, ln=config.ln)

        # classifier
        self.classifier = nn.Linear(in_features=config.hidden_feat, out_features=train_config.num_logits())

        initialize_weights(self)

    def forward(self, data: PatchBatch):
        x = data.fts
        for block in self.gab_blocks:
            x = block(x)

        feat = self.pooling(x)
        logits = self.classifier(feat)

        logits = logits.squeeze(1)

        return logits

# model = ILRA(feat_dim=1024, n_classes=2, hidden_feat=256, num_heads=8, topk=4)
# x = torch.randn((1, 1600, 1024))
