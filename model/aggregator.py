import torch
from torch import nn
from typing import Tuple

import utils


class TransformerAggregator(nn.Module):
    """
    Fairly simple wrapper for nn.Transformer.
    Takes two sequences, (B x N x D) and (B x M x D), with padding for both, and returns aggregated features of
    shape (B x D'). One sequence is the condition, and the other the aggregator input.

    Note: the models presented in this paper do not use the condition sequence, but this may be useful to incorporate
    extra information, such as other modalities.
    """

    def __init__(self, input_dim: int, model_dim: int, output_dim: int, nhead: int, layers: int, dropout: float):
        super().__init__()

        self.dim = model_dim
        self.proj_in = nn.Linear(input_dim, model_dim)
        self.proj_out = nn.Identity()

        self.transformer = nn.Transformer(
            model_dim,
            nhead=nhead,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.special_token = nn.Parameter(torch.randn(model_dim,), requires_grad=True)

    def pos_encode_1d(self, xs, project=True):
        if project:
            xs = self.proj_in(xs)
        batch_size, length, dim = xs.shape
        return xs + utils.positional_encoding(length, dim, device=xs.device)[None]

    def pos_encode_2d(self, data, normalized_locs, project=True):
        """normalized_locs should be patch indexed, e.g. [[0, 0], [0, 1], [1, 0], [3, 1], ...]"""
        # xs: B x S x D
        # locs: B x S x 2
        if project:
            data = self.proj_in(data)
        batch_size, length, dim = data.shape
        assert normalized_locs.shape == (batch_size, length, 2)

        xs = normalized_locs[:, :, 0]
        ys = normalized_locs[:, :, 1]

        enc = utils.positional_encoding_2d_from_pos(xs.view(-1), ys.view(-1), dim, device=xs.device)
        return data + enc.view(batch_size, length, dim)

    def forward(self, seq1, seq2, lengths1, lengths2):
        """seq1 is input to the TransformerEncoder, seq2 to the TransformerDecoder."""
        batch_size, _, _ = seq2.shape

        # Add special token
        special_toks = self.special_token.view(1, 1, -1).repeat((batch_size, 1, 1))  # Reshape to B x 1 x D
        seq2 = torch.cat((special_toks, seq2), dim=1)
        lengths2 = lengths2 + 1  # Update lengths

        mask1 = utils.padding_mask(seq1, lengths1) if lengths1 is not None else None
        mask2 = utils.padding_mask(seq2, lengths2) if lengths2 is not None else None

        out = self.transformer(
            src=seq1, tgt=seq2, src_key_padding_mask=mask1, memory_key_padding_mask=mask1, tgt_key_padding_mask=mask2
        )

        # Extract at special token locations
        agg = out[:, 0]  # B x (M + 1) x D    ->    B x D
        return agg
