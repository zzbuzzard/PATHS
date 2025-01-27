import os
from os.path import join
import torch


def load(preprocessed_root: str, slide_id: str, power: float):
    path = join(preprocessed_root, slide_id + f"_{power:.3f}.pt")
    assert os.path.isfile(path), f"Pre-process load: path '{path}' not found!"
    return torch.load(path)


def get_all_slide_ids(preprocessed_root: str, base_power):
    fnames = os.listdir(preprocessed_root)
    ending = f"_{base_power:.3f}.pt"
    return [i[:-len(ending)] for i in fnames if i.endswith(ending)]
