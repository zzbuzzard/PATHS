"""
The PatchBatch class. A batch of data is quite a complicated object here, containing variable length padded lists of
patches along with their positions, global + local context vectors and hierarchical information. The PatchBatch class
presents a simple interface for this complicated collection of data.
"""
import torch
from typing import Tuple, Dict

from .slide import RawSlide, PreprocessedSlide
import utils


class PatchBatch:
    """
    A batch of single magnification slide data, used at both train and inference time. Contains sanity checks,
    computes the padding mask, and keeps the code clean (rather than passing around these ~7 arguments).

    Essentially handles some of the complexity of processing complex objects of different lengths in one batch.
    """
    def __init__(self,
                 locs: torch.LongTensor,
                 num_ims: torch.LongTensor,
                 parent_inds: torch.LongTensor,
                 ctx_slide: torch.Tensor,
                 ctx_patch: torch.Tensor,
                 fts: torch.Tensor,
                 **unused_kwargs):
        """
        :param locs: locations of each patch feature, in *pixel coordinates at that magnification*. Shape (B x N x 2)
        :param num_ims: number of images in each batch. Shape (B)
        :param parent_inds: parent index (at prev magnification) of each patch. Useful for visualisation. shape (B x N).
        :param ctx_slide: slide-level context (i.e. F^1, F^2, ...). shape (B x Depth x D_slide)
        :param ctx_patch: patch-level hierarchical context. shape (B x N x Depth x D_patch).
          Note: when LSTM is used, this is actually used to store LSTM state rather than patch features.
        :param fts: patch features. Shape (B x N x D). Padded features are the zero vector, but should not be exposed
          to the model at any point.
        """
        batch_size, max_patches, c = fts.shape

        _, self.ctx_depth, self.ctx_dim1 = ctx_slide.shape
        self.ctx_dim2 = ctx_patch.shape[-1]

        # Check all shapes
        assert locs.shape         == (batch_size, max_patches, 2)
        assert num_ims.shape      == (batch_size,)
        assert parent_inds.shape  == (batch_size, max_patches)
        assert ctx_slide.shape    == (batch_size, self.ctx_depth, self.ctx_dim1)
        assert ctx_patch.shape    == (batch_size, max_patches, self.ctx_depth, self.ctx_dim2)

        assert num_ims.max().item() == max_patches

        # Obtain and check device
        self.device = fts.device
        assert self.device == locs.device == num_ims.device == parent_inds.device == ctx_slide.device == ctx_patch.device == fts.device

        self.batch_size = batch_size
        self.max_patches = max_patches

        self.fts = fts
        self.locs = locs
        self.num_ims = num_ims
        self.parent_inds = parent_inds
        self.ctx_slide = ctx_slide
        self.ctx_patch = ctx_patch

        # Create indices which are in range using num_ims
        inds = torch.arange(max_patches, device=num_ims.device).expand(batch_size, -1)
        inds = inds < num_ims[:, None]
        # Now self.patches[valid_inds] will extract only non-padding patches
        self.valid_inds = inds


def from_batch(batch: Dict, device) -> PatchBatch:
    batch = {i: utils.todevice(j, device) for i, j in batch.items()}
    return PatchBatch(**batch)


def from_raw_slide(slide: RawSlide, im_enc, transform, device=None) -> PatchBatch:
    """
    Creates a PatchBatch object from a RawSlide + Image Encoder, ready to be input to the model.

    Note: carries out image preprocessing, and patch loading if not loaded yet. All patches are encoded as a single
    batch, as we assume K is sufficiently low to allow this.
    """
    if device is None:
        device = utils.device

    # Helper function: add singleton batch dim, move to cuda
    def p(x):
        if x is None:
            return x
        if isinstance(x, tuple) or isinstance(x, list):
            return [p(i) for i in x]
        return x[None].to(device)

    if slide.patches is None:
        slide.load_patches()
    with torch.no_grad():
        fts = im_enc(transform(slide.patches.to(device)))

    num_ims = torch.LongTensor([slide.locs.size(0)]).to(device)
    return PatchBatch(p(slide.locs), num_ims, p(slide.parent_inds), p(slide.ctx_slide), p(slide.ctx_patch), p(fts))


def from_preprocessed_slide(slide: PreprocessedSlide, device=None) -> PatchBatch:
    """
    Creates a PatchBatch object from a PreprocessedSlide. As the patches are preprocessed, there is no need for an image
    encoder.
    """
    if device is None:
        device = utils.device

    # Helper function: add singleton batch dim, move to cuda
    def p(x):
        if x is None:
            return x
        if isinstance(x, tuple) or isinstance(x, list):
            return [p(i) for i in x]
        return x[None].to(device)

    num_ims = torch.LongTensor([slide.locs.size(0)]).to(device)
    fts = slide.fts[0]
    return PatchBatch(p(slide.locs), num_ims, p(slide.parent_inds), p(slide.ctx_slide), p(slide.ctx_patch), p(fts))
