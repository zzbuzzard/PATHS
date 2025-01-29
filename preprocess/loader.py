import os
from os.path import join
import torch

# cache loads to speed up multi-seed running
_DICT = {}


def load(preprocessed_root: str, slide_id: str, power: float):
    path = join(preprocessed_root, slide_id + f"_{power:.3f}.pt")
    assert os.path.isfile(path), f"Pre-process load: path '{path}' not found!"

    if path in _DICT:
        return _DICT[path]

    a = torch.load(path)
    _DICT[path] = a
    return a


def get_all_slide_ids(preprocessed_root: str, base_power):
    fnames = os.listdir(preprocessed_root)
    ending = f"_{base_power:.3f}.pt"
    return [i[:-len(ending)] for i in fnames if i.endswith(ending)]
