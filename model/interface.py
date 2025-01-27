"""Defines an abstract class which models implement (and also the LSTM)."""
import torch
from torch import nn
from abc import abstractmethod, ABC
from typing import Tuple, Dict, Callable

from data_utils.patch_batch import PatchBatch


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        """
        :param input_dim: dimension of input (x)
        :param output_dim: dimension of state / output (h)
        :param hidden_dim: dimension of memory vector (c)
        """
        super().__init__()
        self.xdim = input_dim
        self.hdim = output_dim
        self.cdim = hidden_dim

        # For modifying the memory c
        self.forget_gate = nn.Sequential(nn.Linear(self.xdim + self.hdim, self.cdim), nn.Sigmoid())
        self.remember_gate = nn.Sequential(nn.Linear(self.xdim + self.hdim, self.cdim), nn.Sigmoid())
        self.remember_map = nn.Sequential(nn.Linear(self.xdim + self.hdim, self.cdim), nn.Tanh())

        # For producing the output h(t) from h(t-1), c(t), x(t)
        self.out_select_gate = nn.Sequential(nn.Linear(self.hdim + self.xdim, self.hdim), nn.Sigmoid())
        self.mem_to_out = nn.Sequential(nn.Linear(self.cdim, self.hdim), nn.Tanh())

    def forward(self, xs, hs, cs):
        """
        Carries out the action of a single LSTM cell. Note that the input is not a sequence, but the previous LSTM
        state, and the function returns the next LSTM state.

        :param xs: Inputs at step t. Shape (... x xdim)
        :param hs: Outputs at step t. Shape (... x hdim)
        :param cs: Memories at step t. Shape (... x cdim)
        :return: (hs(t+1), cs(t+1))
        """
        *b1, xdim = xs.shape
        *b2, hdim = hs.shape
        *b3, cdim = cs.shape
        assert b1 == b2 == b3, f"Mismatching starting dimensions: ({b1}, {b2}, {b3}) should be identical."
        assert xdim == self.xdim, f"Input dim {xdim} but {self.xdim} expected."
        assert hdim == self.hdim, f"Output dim {hdim} but {self.hdim} expected."
        assert cdim == self.cdim, f"Memory dim {cdim} but {self.cdim} expected."

        xhs = torch.cat((xs, hs), dim=-1)

        # Update memory
        cs = cs * self.forget_gate(xhs)
        cs = cs + self.remember_gate(xhs) * self.remember_map(xhs)

        # Update output
        hs = self.out_select_gate(xhs) * self.mem_to_out(cs)

        return hs, cs


class Processor(ABC):
    """Abstract class for a single-depth slide processor."""

    @abstractmethod
    def process(self, data: PatchBatch) -> Dict:
        """
        Processes a batch of data, returns
        {
            "logits": classification logits,
            "ctx_slide": per-slide context vectors,
            "ctx_patch": per-patch context vectors,
            "importance": per-patch importance scores
        }
        """
        raise NotImplementedError

    @abstractmethod
    def ctx_dim(self) -> Tuple[int, int]:
        """Dimensionality of ctx vectors, (slide ctx dim, patch ctx dim)"""
        raise NotImplementedError


class RecursiveModel(nn.Module):
    """A simple wrapper for N Processors"""
    def __init__(self, processor_constructor: Callable, config_, train_config, **kwargs):
        super().__init__()
        self.procs = nn.ModuleList([processor_constructor(config_, train_config, depth=i, **kwargs) for i in range(train_config.num_levels)])

        # Define LSTM here so it is shared between the processors
        from config import PATHSProcessorConfig
        if isinstance(config_, PATHSProcessorConfig) and config_.lstm:
            dim = config_.patch_embed_dim if config_.model_dim is None else config_.model_dim

            self.lstm = LSTMCell(dim, dim, config_.hierarchical_ctx_mlp_hidden_dim)
            self.use_lstm = True
        else:
            self.use_lstm = False

    def forward(self, depth, *args, **kwargs):
        if self.use_lstm:
            kwargs["lstm"] = self.lstm
        return self.procs[depth].process(*args, **kwargs)
