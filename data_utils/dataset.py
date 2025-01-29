import numpy as np
import torch.utils.data as dutils
from torch.utils.data.dataloader import default_collate
from torch.nn.functional import pad
from torch.multiprocessing import Pool, cpu_count
import pandas as pd
import os
from os.path import join
import torch
from tqdm import tqdm
from typing import List, Tuple
import csv

import utils
from .slide import load_patch_preprocessed_slide
from preprocess.loader import get_all_slide_ids


def multi_dataset_frame(config) -> pd.DataFrame:
    """Create a DataFrame for multi-dataset classification, containing slide IDs + the dataset they came from."""
    s = config.preprocess_dir
    assert config.multi_dataset[0] in s

    # here for clarity but doesn't particularly matter; we are just assigning a unique ID to each dataset
    def get_oncotree_code(ds: str) -> str:
        ds = ds.lower()
        if ds == "kirp":
            return "PRCC"
        elif ds == "kirc":
            return "CCRCC"
        elif ds == "kich":
            return "CHRCC"
        return ds.upper()  # e.g. LUAD, LUSC

    get_preprocess_dir = lambda s: config.preprocess_dir.replace(config.multi_dataset[0], s)

    frame = pd.DataFrame(columns=["slide_id", "oncotree_code", "root_dir"])

    for ds in config.multi_dataset:
        pdir = get_preprocess_dir(ds)
        slide_ids = get_all_slide_ids(pdir, config.base_power)
        code = [get_oncotree_code(ds)] * len(slide_ids)
        root_dir = [pdir] * len(slide_ids)

        temp_frame = pd.DataFrame({"slide_id": slide_ids, "oncotree_code": code, "root_dir": root_dir})
        frame = pd.concat([frame, temp_frame], ignore_index=True)

    return frame


# Returns a tuple: the train/val/test datasets. val may be None.
def load_splits(props, seed, ctx_dim, config, test_only=False, combined=False):
    train_prop, val_prop, test_prop = props
    assert abs(train_prop + val_prop + test_prop - 1) < 1e-4

    if config.multi_dataset is not None:
        frame = multi_dataset_frame(config)
    else:
        # Load CSV here and split the dataset
        frame = pd.read_csv(config.csv_path, compression="zip")
        frame["root_dir"] = config.preprocess_dir  # required to support multi-dataset mode

        # remove .svs extension
        frame["slide_id"] = [".".join(s.split(".")[:-1]) for s in frame["slide_id"]]

    # Prune invalid rows with no corresponding slide
    invalid_labels = []
    for i in range(len(frame)):
        slide_id = frame.iloc[i].slide_id
        root_dir = frame.iloc[i].root_dir
        path = os.path.join(root_dir, slide_id + f"_{config.base_power:.3f}.pt")

        if not os.path.isfile(path):
            invalid_labels.append(i)

    print(f"Ignoring {len(invalid_labels)} rows without files.")
    frame.drop(invalid_labels, inplace=True)

    if config.task == "survival":
        # Select one WSI per patient for survival
        frame = frame.drop_duplicates(subset='case_id')

    frame.reset_index(drop=True, inplace=True)

    # Filter to necessary columns
    if config.task == "survival":
        frame = frame[["case_id", "slide_id", "survival_months", "censorship", "oncotree_code", "root_dir"]]
        _, bins = pd.qcut(frame.survival_months, config.nbins, labels=False, retbins=True)
    else:
        frame = frame[["slide_id", "oncotree_code", "root_dir"]]
        bins = None

    if config.filter_to_subtypes is not None:
        frame = frame[frame['oncotree_code'].isin(config.filter_to_subtypes)]

    if combined:
        return SlideDataset(frame, bins, ctx_dim, config)

    if config.hipt_splits:
        ds = os.path.split(config.wsi_dir)[-1].lower()  # e.g. "brca"

        if config.task == "survival":
            path = f"data/splits/survival/tcga_{ds}"
        elif config.task == "subtype_classification":
            name = ds
            if ds == "kirp": name = "kidney"
            if ds == "luad": name = "lung"
            path = f"data/splits/subtype_classification/tcga_{name}"
        else:
            raise Exception(f"Unexpected task '{config.task}' - expected subtype_classification or survival.")

        if not os.path.isdir(path):
            print(f"Error: couldn't find path {path}")
            quit(1)
        path = os.path.join(path, f"splits_{seed}.csv")

        if config.task == "subtype_classification":
            with open(path, "r") as f:
                r = csv.reader(f)
                next(r)  # remove column titles
                data = [i[1:] for i in r]
            train_p = [i for i, j, k in data]
            val_p = [j for i, j, k in data if len(j) > 0]
            test_p = [k for i, j, k in data if len(k) > 0]
            match_on = 'slide_id'
        else:
            with open(path, "r") as f:
                r = csv.reader(f)
                next(r)  # remove column titles
                data = [i[1:] for i in r]
            train_p = [i for i, j in data]
            val_p = None
            test_p = [j for i, j in data if len(j) > 0]
            match_on = 'case_id'

            if config.hipt_val_proportion > 0:
                val_size = int(len(train_p) * config.hipt_val_proportion)
                val_p, train_p = train_p[:val_size], train_p[val_size:]

        train = frame[frame[match_on].isin(train_p)]
        val = frame[frame[match_on].isin(val_p)] if val_p is not None else None
        test = frame[frame[match_on].isin(test_p)]

        print(f"HIPT split: {len(train)}/{len(val) if val is not None else 0}/{len(test)}")
    else:
        # Randomly sample train/val/test datasets
        train_c = int(train_prop * len(frame))
        val_c = int(val_prop * len(frame))
        test_c = len(frame) - train_c - val_c
        print(f"Partitioning dataset of {len(frame)} into {train_c}/{val_c}/{test_c} items.")

        train = frame.sample(train_c, random_state=seed)
        val = frame.drop(train.index).sample(val_c, random_state=seed)
        test = frame.drop(train.index).drop(val.index)

    if test_only:
        test.reset_index(inplace=True, drop=True)
        return SlideDataset(test, bins, ctx_dim, config)

    ds = []
    for frame in [train, val, test]:
        if frame is None:
            ds.append(None)
        else:
            frame.reset_index(inplace=True, drop=True)
            ds.append(SlideDataset(frame, bins, ctx_dim, config))

    return ds


class SlideDataset(dutils.Dataset):
    """
    Dataset of PreprocessedSlides. Also stores metadata such as survival/censorship info.
    """

    def __init__(self, frame: pd.DataFrame, bins, ctx_dim, config):
        super().__init__()
        self.wsi_dir = config.wsi_dir
        self.patch_size = config.model_config.patch_size
        self.base_power = config.base_power
        self.num_levels = config.num_levels

        # M=4 is implemented at the data level as M=2 but with double recursion
        #  so for dataset purposes, we have twice as many levels (where a level has a fixed increase of 2x)
        if config.magnification_factor == 4:
            # e.g. 0.625, 3 levels, M=4 : we load patches at [0.625, 1.25, 2.5, 5.0, 10.0] = 5 levels
            self.num_levels = 2 * self.num_levels - 1

        self.slide_ids = frame.slide_id
        self.root_dirs = frame.root_dir.tolist()

        ds_len = len(self.slide_ids)

        self.ctx_dim = ctx_dim

        if config.task == "subtype_classification":
            classes = frame.oncotree_code.tolist()
            subtypes = sorted(set(classes))

            # len(subtypes) may occasionally be != config.num_logits() for e.g. small val sets
            print("Distinct subtypes:", len(subtypes))
            print("Subtype counts:")
            for i in subtypes:
                c = classes.count(i)
                print(f" {i}:\t\t{c}")

            self.subtype = [subtypes.index(i) for i in classes]

            self.q_survival_months = None
            self.survival_months = None
            self.censorship = None
        else:
            self.q_survival_months = pd.cut(frame.survival_months, bins, labels=False, include_lowest=True)
            self.survival_months = frame.survival_months
            self.censorship = torch.tensor(frame.censorship.to_numpy(np.int64), dtype=torch.long)

            self.subtype = None

        # Single-threaded version
        self.slides = []
        for i in tqdm(range(ds_len), desc="Pre-patching dataset..."):
            self.slides.append(self.load_top_level(i))

        # Multi-threaded version
        # torch.multiprocessing.set_sharing_strategy('file_system')
        # num_workers = min(cpu_count(), utils.MAX_WORKERS)
        # print("Using", num_workers, "workers")
        # with Pool(num_workers) as pool:
        #     inps = list(range(ds_len))
        #     data = list(tqdm(pool.imap(self.load_top_level, inps), total=ds_len, desc="Pre-patching dataset"))
        # self.slides = data
        # torch.multiprocessing.set_sharing_strategy('file_descriptor')

    def load_top_level(self, idx):
        kwargs = {}
        if self.subtype is not None:
            kwargs["subtype"] = self.subtype[idx]

        preprocessed_root = self.root_dirs[idx]
        slide_id = self.slide_ids[idx]
        if slide_id.endswith(".svs"):
            slide_id = slide_id[:-4]

        return load_patch_preprocessed_slide(slide_id, preprocessed_root, self.base_power, self.patch_size,
                                             self.ctx_dim, self.num_levels, **kwargs)

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, item):
        s = self.slides[item]

        # survival mode
        if self.q_survival_months is not None:
            label_data = {
                "survival_bin": self.q_survival_months[item],
                "survival": self.survival_months[item],
                "censored": self.censorship[item]
            }
        # classification mode; subtype is included in s.todict() already
        else:
            label_data = {}

        return s.todict() | label_data | {"slide": s}


def collate_fn(xs):
    """Special collate_fn to pad the fields which have variable length."""
    fts = [i.pop("fts") for i in xs]                  # (variable) x D
    locs = [i.pop("locs") for i in xs]                # (variable) x 2
    ctx_patch = [i.pop("ctx_patch") for i in xs]      # (variable) x K x D
    parent_inds = [i.pop("parent_inds") for i in xs]  # (variable)

    num_ims = [i.shape[0] for i in locs]
    max_ims = max(num_ims)
    num_ims = torch.LongTensor(num_ims)

    fts = torch.cat([pad(i, (0, 0, 0, max_ims - i.shape[0]))[None] for i in fts])
    locs = torch.cat([pad(i, (0, 0, 0, max_ims - i.shape[0]))[None] for i in locs])
    parent_inds = torch.cat([pad(i, (0, max_ims - i.shape[0]))[None] for i in parent_inds])

    # Annoyingly, the pad function crashes when presented with tensors of shape (N, 0, D)
    # So here's a workaround
    _, k, d = ctx_patch[0].shape
    if k == 0:
        ctx_patch = torch.zeros((locs.size(0), max_ims, 0, d), dtype=ctx_patch[0].dtype, device=ctx_patch[0].device)
    else:
        ctx_patch = torch.cat([pad(i, (0, 0, 0, 0, 0, max_ims - i.shape[0]))[None] for i in ctx_patch])

    padded_data = {
        "fts": fts,
        "locs": locs,                # B x MaxIms x 2
        "ctx_patch": ctx_patch,      # B x MaxIms x K x D
        "parent_inds": parent_inds,  # B x MaxIms
        "num_ims": num_ims,          # B
    }

    # `slide` is included by the dataset, but not during recursion (see `PreprocessedSlide.iter`)
    if "slide" in xs[0].keys():
        extra = {"slide": [i.pop("slide") for i in xs]}
    else:
        extra = {}

    return default_collate(xs) | padded_data | extra
