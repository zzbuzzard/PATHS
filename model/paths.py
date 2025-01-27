import torch
from torch import nn
from typing import Dict, Tuple

from data_utils.patch_batch import PatchBatch
from .interface import Processor
from .aggregator import TransformerAggregator
import config as cfg
import utils


class PATHSProcessor(nn.Module, Processor):
    """
    Implementation of a processor for one magnification level, Pi, implementing the abstract class Processor.
    The full model is an `interface.RecursiveModel` containing several `PATHSProcessor`s.
    """
    def __init__(self, config, train_config, depth: int):
        super().__init__()
        train_config: cfg.Config
        config: cfg.PATHSProcessorConfig

        self.depth = depth

        # Output dimensionality
        num_logits = train_config.num_logits()

        self.config = config
        self.train_config = train_config

        if config.model_dim is None:
            self.proj_in = nn.Identity()
            self.dim = config.patch_embed_dim
        else:
            self.proj_in = nn.Linear(config.patch_embed_dim, config.model_dim, bias=False)
            self.dim = config.model_dim

        self.slide_ctx_dim = config.trans_dim

        # Slide context can either be concatenated or summed; in our paper we choose sum (mode="residual")
        if self.config.slide_ctx_mode == "concat":
            self.classification_layer = nn.Linear(self.slide_ctx_dim * (depth + 1), num_logits)
        else:
            self.classification_layer = nn.Linear(self.slide_ctx_dim, num_logits)

        # Per-patch MLP to produce importance values \alpha
        self.importance_mlp = nn.Sequential(
            nn.Linear(self.dim, config.importance_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.importance_mlp_hidden_dim, 1),
        )

        if config.lstm:
            self.hdim = config.hierarchical_ctx_mlp_hidden_dim
        else:
            # A simple RNN instead of the LSTM
            self.hctx_mlp = nn.Sequential(
                nn.Linear(self.dim, config.hierarchical_ctx_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hierarchical_ctx_mlp_hidden_dim, self.dim)
            )

        # Global aggregator
        self.global_agg = TransformerAggregator(
            input_dim=self.dim,
            model_dim=config.trans_dim,
            output_dim=self.dim,
            nhead=config.trans_heads,
            layers=config.trans_layers,
            dropout=config.dropout,
        )

    def process(self, data: PatchBatch, lstm=None) -> Dict:
        """
        Process a batch of slides to produce a batch of logits. The class PatchBatch is defined to manage the complexity
        of each slide having different length etc. (padding is required).
        """
        patch_features = data.fts
        patch_features = self.proj_in(patch_features)

        ################# Apply LSTM
        if self.config.lstm:
            assert lstm is not None

            # Initialise LSTM state at top of hierarchy
            if self.depth == 0:
                hs = torch.zeros((data.batch_size, data.max_patches, self.dim), device=data.device)
                cs = torch.zeros((data.batch_size, data.max_patches, self.hdim), device=data.device)

            # Otherwise, retrieve it
            else:
                lstm_state = data.ctx_patch[:, :, -1]
                assert lstm_state.shape[-1] == self.dim + self.hdim
                hs, cs = lstm_state[..., :self.dim], lstm_state[..., self.dim:]

            hs, cs = lstm(patch_features, hs, cs)
            patch_features = patch_features + hs  # produce Y from X

            patch_ctx = torch.concat((hs, cs), dim=-1)

        ################# Get importance values \alpha
        # (this method ensures padding is assigned 0 importance: apply the MLP+sigmoid only to non-background patches)
        importance = utils.apply_to_non_padded(lambda xs: torch.sigmoid(self.importance_mlp(xs)), patch_features, data.valid_inds, 1)[..., 0]
        if self.config.importance_mode == "mul":
            # produce Z from Y
            patch_features = patch_features * importance[..., None]

        # If not using the LSTM, apply a RNN instead
        if not self.config.lstm:
            if self.depth > 0 and self.config.hierarchical_ctx:
                assert len(data.ctx_patch.shape) == 4
                hctx = data.ctx_patch[:, :, -1]  # B x MaxIms x D
                hctx = utils.apply_to_non_padded(self.hctx_mlp, hctx, data.valid_inds, self.dim)

                patch_features = patch_features + hctx

            patch_ctx = patch_features

        ################# Global aggregation
        d = self.config.trans_dim

        # Unused conditional sequence for aggregation. We tried putting slide context here but it performed poorly
        #  compared to residual context.
        encoder_input = torch.zeros((data.batch_size, 0, d), device=data.device)

        # Positional encoding
        xs = patch_features
        patch_locs = data.locs // self.config.patch_size  # pixel coords -> patch coords
        if self.config.pos_encoding_mode == "1d":
            xs = self.global_agg.pos_encode_1d(xs)
        elif self.config.pos_encoding_mode == "2d":
            xs = self.global_agg.pos_encode_2d(xs, patch_locs)

        # Aggregate
        slide_features = self.global_agg(encoder_input, xs, None, data.num_ims)

        # Apply residual connection
        if self.config.slide_ctx_mode == "residual" and data.ctx_depth > 0:
            slide_features = slide_features + data.ctx_slide[:, -1]

        ################# Produce final logits
        if self.config.slide_ctx_mode == "concat":
            all_ctx = torch.flatten(data.ctx_slide, start_dim=1)  # B x K x D -> B x (K * D)
            ft = torch.cat((all_ctx, slide_features), dim=1)
            logits = self.classification_layer(ft)
        else:
            logits = self.classification_layer(slide_features)

        return {
            "logits": logits,
            "ctx_slide": slide_features,
            "ctx_patch": patch_ctx,  # (actually RNN state)
            "importance": importance
        }

    def ctx_dim(self) -> Tuple[int, int]:
        if self.config.lstm:
            return self.slide_ctx_dim, self.dim + self.hdim
        return self.slide_ctx_dim, self.dim
