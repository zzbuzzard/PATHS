import torch
from torch import nn

from data_utils.patch_batch import PatchBatch
import utils


class ABMIL(nn.Module):
    def __init__(self, config, train_config):
        super().__init__()
        self.dim = config.patch_embed_dim
        num_logits = train_config.nbins if train_config.task == "survival" else len(train_config.filter_to_subtypes)
        self.gate1 = torch.nn.Sequential(torch.nn.Linear(self.dim, 1, bias=False), torch.nn.Tanh()).eval()
        self.gate2 = torch.nn.Sequential(torch.nn.Linear(self.dim, 1, bias=False), torch.nn.Sigmoid()).eval()
        self.final_project = torch.nn.Linear(self.dim, num_logits, bias=False)

    def new_depth(self, new_depth: int):
        pass

    def forward(self, data: PatchBatch):
        xs = data.fts                               # B x N x D
        a1 = utils.apply_to_non_padded(self.gate1, xs, data.valid_inds, 1)  # B x N x 1
        a2 = utils.apply_to_non_padded(self.gate2, xs, data.valid_inds, 1)  # B x N x 1
        a = a1 * a2  # B x N x 1

        # Each batch item has a different number of features.
        # Here we set padding locations to a large negative number before softmax,
        #  ensuring they are assigned an importance of (practically) 0
        a[~data.valid_inds] = -10000

        a = torch.softmax(a[..., 0], dim=-1)        # B x N
        z = torch.sum(a[..., None] * xs, dim=1)     # B x N x D -> B x D
        res = self.final_project(z)

        return res
