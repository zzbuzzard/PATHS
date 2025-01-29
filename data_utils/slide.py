"""
This file contains two Slide classes: PreprocessedSlide, for WSIs which have been prepatched and processed with an image
encoder, and RawSlide, for WSIs which we wish to patch on-the-fly (at inference time).

The classes store hierarchical information, handle the patches and their location, hierarchical context, etc.
"""
import torch
import torchvision.transforms.functional as trf
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
from typing import Tuple
import numpy as np
import os

from preprocess import loader
import utils


def camelyon_map(patch):
    img = patch.copy()
    black_background = np.mean(img, axis=2) <= 0.01 * 255
    img[black_background] = 255
    return img


class RawSlide:
    """
    A raw (not preprocessed) slide. Can be used in conjunction with an image encoder to process new slides at inference
    time. For examples, see visualise.py.

    Note: only used at inference, not during training, as preprocessing patches is a huge speedup.
    """
    def __init__(self, path: str, power: float, patch_size: int, load_locs: torch.LongTensor, load_size: Tuple[int, int],
                 ctx_slide: torch.Tensor, parent_ctx_patch: torch.Tensor, tissue_threshold: float = 0.1,
                 ctx_patch_dim: int = None, keep_inds=None, subtype=None):
        self.path = path
        self.patch_size = patch_size
        self.power = power
        self.load_locs = load_locs  # the pixel position at this resolution for *loads* which will then be patched
        self.ctx_slide = ctx_slide
        self.parent_ctx_patch = parent_ctx_patch
        self.tissue_threshold = tissue_threshold
        self.load_size = load_size
        self.ctx_patch_dim = ctx_patch_dim
        self.keep_inds = keep_inds
        self.subtype = subtype

        # None until load_patches() is called
        self.patches = None
        self.locs = None
        self.parent_inds = None
        self.ctx_patch = None
        self.size_pixels = None  # total slide size at this power in pixels

        self.camelyon = False  # used for visualisation only

    def parent_ind_map(self):
        """Returns a map from my patch indices to the indices in my parent slide"""
        return self.keep_inds[self.parent_inds]

    def unload_patches(self):
        self.patches = self.locs = self.parent_inds = self.ctx_patch = None

    def view_at_power(self, power):
        wsi = WSIReader.open(self.path)
        if wsi.info.objective_power is None:
            print("No objective power; assuming 40")
            wsi._m_info.objective_power = 40
        ht, wt = wsi.slide_dimensions(resolution=power, units="power")
        out = wsi.read_rect(
            (0, 0),
            (ht, wt),
            resolution=power,
            units="power"
        )
        if self.camelyon:
            out = camelyon_map(out)
        return out

    def load_patches(self, wsi=None):
        if self.patches is not None:
            print("WARNING: Trying to load_patches() but they have already been loaded.")
            return

        h, w = self.load_size
        assert h % self.patch_size == 0 and w % self.patch_size == 0, f"Load size {self.load_size} is not divisible by patch size {self.patch_size}."

        if wsi is None:
            wsi = WSIReader.open(self.path)
            if wsi.info.objective_power is None:
                print("No objective power; assuming 40")
                wsi._m_info.objective_power = 40

        ht, wt = wsi.slide_dimensions(resolution=self.power, units="power")
        ht, wt = utils.next_multiple(ht, self.patch_size), utils.next_multiple(wt, self.patch_size)
        self.size_pixels = (ht, wt)

        masker = OtsuTissueMasker()

        # Read in all locs
        ims = []
        for loc in self.load_locs:
            y, x = loc.tolist()

            im = wsi.read_rect(
                (x, y),
                self.load_size,
                resolution=self.power,
                units="power",
                coord_space="resolution"  # note that loc is in pixel space for power=self.power
            )
            if self.camelyon:
                im = camelyon_map(im)

            ims.append(im)

        # this is a bit of a workaround as tiatoolbox doesn't appear to have a close() method
        # necessary to prevent error due to excessive open files
        wsi.openslide_wsi.close()

        # Produce parent_inds, which gives the index in load_locs that each *patch* came from
        num_images = (h // self.patch_size) * (w // self.patch_size)
        parent_inds = torch.tensor(sum([[i] * num_images for i in range(self.load_locs.shape[0])], []),
                                      dtype=torch.long, requires_grad=False)

        masks = masker.fit_transform(ims)

        # Concatenate loads
        ims = torch.cat([trf.to_tensor(i)[None] for i in ims])
        masks = torch.cat([trf.to_tensor(m)[None] for m in masks])

        # Patchify loads
        # Now patches: B x NumImages x C x H x W, locs: B x NumImages x 2
        patches, locs = utils.patchify_locs(ims, self.patch_size, self.load_locs)
        mask_patches = utils.patchify(masks, self.patch_size, channels=1)

        # Flatten first dims; it is no longer important what was loaded from where (this is tracked by parent_inds)
        locs = locs.flatten(end_dim=1)
        patches = patches.flatten(end_dim=1)
        mask_patches = mask_patches.flatten(end_dim=1)

        # True indicates tissue, False indicates background, so tissue_proportions=1 means all tissue
        tissue_proportions = torch.sum(mask_patches, dim=(1, 2, 3)) / (self.patch_size * self.patch_size)

        # Keep only patches with sufficient tissue
        threshold = self.tissue_threshold
        indices = tissue_proportions > threshold
        while torch.sum(indices).item() == 0 and threshold > 1e-6:
            print(f"Oh dear... image has 0 patches with thresh {threshold}, path={self.path}")
            threshold /= 2
            indices = tissue_proportions > threshold

        # Super rare case where everything ends up masked out
        if threshold <= 1e-6:
            print("Everything is masked out!")
            indices = torch.LongTensor([0]).to(indices.device)

        # Apply background filter
        self.patches = patches[indices]             # (N x C x H x W)
        self.locs = locs[indices]                   # (N x 2)
        self.parent_inds = parent_inds[indices]  # (N)

        # Obtain ctx_patch by indexing into parent ctx_patch
        if self.parent_ctx_patch is None:
            n = self.patches.size(0)
            self.ctx_patch = torch.zeros((n, 0, self.ctx_patch_dim))
        else:
            self.ctx_patch = self.parent_ctx_patch[self.parent_inds]

        # retain_patches = indices.sum().item()
        # total_patches = indices.shape[0]

    def recurse(self, multiplier: int, ctx_slide, ctx_patch, importance, keep_patches: int = -1):
        assert len(importance.shape) == 1, f"Invalid shape {importance.shape}"
        if self.patches is None:
            raise Exception("Slide.recurse() called before load_patches()")

        ctx_slide = torch.cat((self.ctx_slide, ctx_slide[None]), dim=0)  # K x D -> (K+1) x D
        ctx_patch = torch.cat((self.ctx_patch, ctx_patch[:, None]), dim=1)  # N x K x D -> N x (K+1) x D

        keep_locs = self.locs

        if keep_patches != -1:
            # Filter by importance
            count = min(importance.size(0), keep_patches)
            keep_inds = torch.topk(importance, count).indices

            ctx_patch = ctx_patch[keep_inds]
            keep_locs = keep_locs[keep_inds]
        else:
            keep_inds = torch.LongTensor(list(range(importance.size(0)))).to(importance.device)

        # Convert from pixel coords at this depth -> pixel coords at the next depth
        load_locs = keep_locs * multiplier
        load_size = (self.patch_size * multiplier, self.patch_size * multiplier)

        return RawSlide(self.path, self.power * multiplier, self.patch_size, load_locs, load_size, ctx_slide, ctx_patch,
                        tissue_threshold=self.tissue_threshold, keep_inds=keep_inds, subtype=self.subtype)

    def todict(self):
        kwargs = {}
        if self.subtype is not None:
            kwargs["subtype"] = self.subtype

        patches = self.patches
        locs = self.locs
        parent_inds = self.parent_inds
        ctx_patch = self.ctx_patch

        return {
            # Variable length
            "patches": patches,
            "locs": locs,
            "parent_inds": parent_inds,
            "ctx_patch": ctx_patch,

            # Fixed length
            "ctx_slide": self.ctx_slide
        } | kwargs

    def __repr__(self):
        npatches = "?" if self.patches is None else self.patches.size(0)
        ctx_depth = self.ctx_slide.size(0)
        return f"Slide(num_patches={npatches}, ctx_depth={ctx_depth}, power={self.power})"


class PreprocessedSlide:
    """
    Stores the preprocessed patches for a slide across all magnifications. Note that, if stored for all slides in a
    dataset, this requires a fair bit of RAM.

    This is the version stored in the Datasets which are used at train time, but is not applicable to 'new' slides
    which have not been preprocessed.
    The preprocessed patches are lodaed according to `preprocess.loader`, which must be initialised with the path
    to the directory containing preprocessed patches (across all magnifications).
    """

    def __init__(self, slide_id: str, preprocessed_root: str, base_power: float, num_levels: int, patch_size: int,
                 ctx_slide: torch.Tensor, ctx_patch_dim: int = None, subtype=None, wsi_root: str = None):
        self.patch_size = patch_size
        self.base_power = base_power
        self.ctx_slide = ctx_slide
        self.ctx_patch_dim = ctx_patch_dim
        self.subtype = subtype

        self.preprocessed_root = preprocessed_root
        self.slide_id = slide_id
        self.wsi_root = wsi_root  # not needed, but may be useful for visualisations

        self.fts = []

        for i in range(num_levels):
            power = base_power * 2**i
            i_fts = loader.load(self.preprocessed_root, self.slide_id, power)

            self.fts.append(i_fts)

        # Filter the first set of features to non-backgorund patches
        fts0 = self.fts[0]

        x, y, _ = fts0.shape
        locs = torch.LongTensor([[i, j] for i in range(x) for j in range(y)])

        ctx_patch = torch.zeros((locs.shape[0], 0, ctx_patch_dim))
        parent_inds = torch.LongTensor(list(range(locs.shape[0])))

        self.locs = locs
        self.ctx_patch = ctx_patch
        self.parent_inds = parent_inds
        self.fts[0] = fts0[locs[:, 0], locs[:, 1]]

        self.size_pixels = None  # total slide size at this power in pixels

    def load_patches(self, wsi=None):
        assert wsi is None
        return

    def iter(self, magnification_index: int, npatches: int, locs, ctx_slide, ctx_patch, importance, new_ctx_slide,
             new_ctx_patch, keep_patches: int = -1, imp_cpu=None):
        """Iterates data from magnification level `index` to `index+1`"""
        locs //= self.patch_size

        if imp_cpu is None and importance is not None:
            imp_cpu = importance.cpu()

        # Remove padding
        ctx_patch = ctx_patch[:npatches]
        if new_ctx_patch is not None:
            new_ctx_patch = new_ctx_patch[:npatches]
        locs = locs[:npatches]
        if imp_cpu is not None:
            imp_cpu = imp_cpu[:npatches]

        if new_ctx_slide is not None:
            ctx_slide = torch.cat((ctx_slide, new_ctx_slide[None]), dim=0)  # K x D -> (K+1) x D
        if new_ctx_patch is not None:
            ctx_patch = torch.cat((ctx_patch, new_ctx_patch[:, None]), dim=1)  # N x K x D -> N x (K+1) x D

        if keep_patches != -1:
            # Filter by importance
            count = min(imp_cpu.size(0), keep_patches)

            keep_inds = torch.topk(imp_cpu, count).indices

            ctx_patch = ctx_patch[keep_inds]
            locs = locs[keep_inds]

        kwargs = {"dtype": locs.dtype, "device": locs.device}

        # Expand each loc into 4: (x, y) -> [(2x, 2y), (2x, 2y+1), (2x+1, 2y), (2x+1, 2y+1)]
        #  corresponding to the index of that location at the subsequent magnification
        new_locs = locs * torch.tensor([2, 2], **kwargs)

        # if parent_inds[i] = parent_inds[j], they share an immediate parent.
        # used with keep_inds, we can reconstruct the whole hierarchy (only needed for visualisation).
        parent_inds = torch.tensor(list(range(new_locs.shape[0])) * 4, **kwargs)   # 0 1 2 ... 0 1 2 ... 0 1 2 ... 0 1 2 ...
        new_locs = torch.cat((new_locs,
                              new_locs + torch.tensor([0, 1], **kwargs),
                              new_locs + torch.tensor([1, 0], **kwargs),
                              new_locs + torch.tensor([1, 1], **kwargs)), dim=0)

        # (2x, 2y), (2x, 2y+1) etc should all have identical ctx_patch
        ctx_patch = torch.cat((ctx_patch,) * 4, dim=0)

        fts = self.fts[magnification_index + 1]
        x, y, _ = fts.shape
        filter_bound = torch.logical_and(new_locs[:, 0] < x, new_locs[:, 1] < y)
        new_locs[~filter_bound] *= 0  # prevent indexerror on next line (modification doesnt matter as they're about to be filtered out)
        filter_bg = fts[new_locs[:, 0], new_locs[:, 1]].sum(dim=1) != 0
        filter = torch.logical_and(filter_bound, filter_bg)

        new_locs = new_locs[filter]
        parent_inds = parent_inds[filter]
        ctx_patch = ctx_patch[filter]

        new_fts = fts[new_locs[:, 0], new_locs[:, 1]]

        # Very rare edge case: zero non-background patches at the second or greater level
        #  (occurs in slides with so little tissue that there are 0 patches at the first magnification level)
        #  We handle this by using all patches at this and subsequent levels
        if new_locs.size(0) == 0:
            # 1) Initialise to all patches
            ctx_patch = torch.zeros((x * y, ctx_patch.size(1), ctx_patch.size(2)), device=ctx_patch.device)
            parent_inds = torch.LongTensor(list(range(x * y)))
            new_locs = torch.LongTensor([[i, j] for i in range(x) for j in range(y)])

            # 2) Filter out background patches
            filter = fts[new_locs[:, 0], new_locs[:, 1]].sum(dim=1) != 0
            if filter.count_nonzero() == 0:
                # (or just all patches if they're all background still, with the hope that there will be tissue later)
                # print("Using all locations: count =", x * y)
                filter[:] = True

            new_locs = new_locs[filter]
            parent_inds = parent_inds[filter]
            ctx_patch = ctx_patch[filter]
            new_fts = fts[new_locs[:, 0], new_locs[:, 1]]

        return {
            "fts": new_fts,
            "ctx_patch": ctx_patch,
            "ctx_slide": ctx_slide,
            "locs": new_locs * self.patch_size,
            "parent_inds": parent_inds
        }

    def recurse(self, *args, **kwargs):
        raise NotImplementedError

    def todict(self):
        kwargs = {}
        if self.subtype is not None:
            kwargs["subtype"] = self.subtype

        fts = self.fts[0]
        locs = self.locs
        parent_inds = self.parent_inds
        ctx_patch = self.ctx_patch

        return {
            # Variable length
            "fts": fts,
            "locs": locs * self.patch_size,
            "parent_inds": parent_inds,
            "ctx_patch": ctx_patch,

            # Fixed length
            "ctx_slide": self.ctx_slide
        } | kwargs


def load_patch_preprocessed_slide(slide_id: str, preprocessed_root: str, base_power: float, patch_size: int,
                                  ctx_dim: Tuple[int, int], num_levels: int, subtype=None) -> PreprocessedSlide:
    ctx_slide = torch.zeros((0, ctx_dim[0]))
    slide = PreprocessedSlide(slide_id, preprocessed_root, base_power, num_levels, patch_size, ctx_slide, ctx_dim[1], subtype=subtype)
    return slide


# path is the full path to the WSI
def load_raw_slide(path: str, base_power: float, patch_size: int, ctx_dim: Tuple[int, int], tissue_threshold: float = 0.1,
              prepatch: bool = True, subtype=None) -> RawSlide:
    load_locs = torch.LongTensor([[0, 0]])
    ctx_slide = torch.zeros((0, ctx_dim[0]))

    wsi = WSIReader.open(path)
    if wsi.info.objective_power is None:
        print("No objective power; assuming 40")
        wsi._m_info.objective_power = 40

    h, w = wsi.slide_dimensions(base_power, "power")
    h, w = utils.next_multiple(h, patch_size), utils.next_multiple(w, patch_size)

    slide = RawSlide(path, base_power, patch_size, load_locs, (h, w), ctx_slide, None,
                     tissue_threshold, ctx_patch_dim=ctx_dim[1], subtype=subtype)
    if prepatch:
        slide.load_patches(wsi)

    return slide
