import os
from os.path import join
import torch

root_dir = None


def set_preprocess_dir(root: str):
    global root_dir
    root_dir = root
    assert os.path.isdir(root_dir), f"Preprocessing root directory '{root_dir}' not found!"


def load(slide_id, power: float):
    assert root_dir is not None, f"set_preprocess_dir must be called before load!"
    path = join(root_dir, slide_id + f"_{power:.3f}.pt")
    assert os.path.isfile(path), f"Pre-process load: path '{path}' not found!"
    return torch.load(path)
