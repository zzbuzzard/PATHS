from dataclasses import dataclass
from os import path
import json
from torch.optim.lr_scheduler import ExponentialLR
from typing import List

from data_utils.dataset import load_splits
from model.paths import PATHSProcessor
from model.baselines import ABMIL, TransMIL, ILRA
from model.interface import RecursiveModel


@dataclass
class ModelConfig:
    pass


# Model configuration (model dependent)
@dataclass
class PATHSProcessorConfig(ModelConfig):
    hierarchical_ctx: bool = True
    slide_ctx_mode: str = "residual"  # residual / concat / none

    patch_embed_dim: int = 1024
    model_dim: int = None  # project embeds to model_dim if it is not None
    dropout: float = 0.0
    patch_size: int = 256  # only needed for visualisation etc. and not at train time

    importance_mode: str = "mul"  # mul / none

    trans_dim: int = 192
    trans_heads: int = 4
    trans_layers: int = 2
    pos_encoding_mode: str = "1d"  # 1d / 2d

    importance_mlp_hidden_dim: int = 128
    hierarchical_ctx_mlp_hidden_dim: int = 256
    lstm: bool = True

    random_rec_baseline: bool = False  # random patch selection. just used for ablation


@dataclass
class ABMILConfig(ModelConfig):
    patch_embed_dim: int = 1024
    patch_size: int = 256


@dataclass
class TransMILConfig(ModelConfig):
    patch_embed_dim: int = 1024
    patch_size: int = 256
    transformer_dim: int = 512


@dataclass
class ILRAConfig(ModelConfig):
    patch_embed_dim: int = 1024
    patch_size: int = 256

    num_layers: int = 2
    hidden_feat: int = 256
    num_heads: int = 8
    topk: int = 64  # default in original codebase is 2, but paper suggests 64
    ln: bool = False


# Training stats etc (model independent)
@dataclass
class Config:
    model_config: ModelConfig

    # Recursion related
    base_power: float
    magnification_factor: int
    num_levels: int
    num_epochs: int
    top_k_patches: List[int]   # how many patches to keep at each level; -1 denotes keep all patches

    model_type: str

    # Data
    wsi_dir: str
    csv_path: str
    preprocess_dir: str = None

    # This is a bit of a workaround. The codebase was designed with single datasets in mind,
    #  but kidney and lung classification require multiple datasets (KIRP/KIRC/KICH and LUSC/LUAD respectively)
    #  setting multi_dataset=["kirp", "kirc", "kich"] causes all three datasets to be read.
    multi_dataset: List[str] = None

    nbins: int = 4
    loss: str = "nll"

    task: str = "survival"  # survival / subtype_classification
    filter_to_subtypes: List[str] = None

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1  # measured in batches
    eval_epochs: int = 1
    lr: float = 2e-5
    lr_decay_per_epoch: float = 0.99
    seed: int = 0
    weight_decay: float = 1e-2
    early_stopping: bool = False
    early_stopping_patience: int = 5

    root_name: str = ""  # for tracking multiple folds

    use_mixed_precision: bool = False

    hipt_splits: bool = False
    hipt_val_proportion: float = 0   # Split part of the HIPT training set off into a val set

    @staticmethod
    def load(root_path: str, test_mode: bool = False):
        """
        Loads a Config object from [root_path]/config.json.
        """
        jsonpath = path.join(root_path, "config.json")
        assert path.isdir(root_path), f"Model directory '{root_path}' not found!"
        assert path.isfile(jsonpath), f"config.json not found in directory '{root_path}'."

        with open(jsonpath, "r") as file:
            data = json.loads(file.read())

        if isinstance(data["top_k_patches"], int):
            data["top_k_patches"] = [data["top_k_patches"]] * (data["num_levels"] - 1)

        if isinstance(data["num_epochs"], list):
            data["num_epochs"] = data["num_epochs"][0]

        if isinstance(data["batch_size"], int):
            data["batch_size"] = [data["batch_size"]] * data["num_levels"]

        if data["model_type"] == "PATHS":
            data["model_config"] = PATHSProcessorConfig(**data["model_config"])
            c = data["model_config"]
            if c.lstm:
                assert c.hierarchical_ctx, "If LSTM mode is enabled, hierarchical context must be enabled."
        elif data["model_type"] == "abmil":
            data["model_config"] = ABMILConfig(**data["model_config"])
        elif data["model_type"] == "transmil":
            data["model_config"] = TransMILConfig(**data["model_config"])
        elif data["model_type"].lower() == "ilra":
            data["model_config"] = ILRAConfig(**data["model_config"])
        else:
            raise NotImplementedError(f"Unknown model type '{data['model_type']}'")

        config = Config(**data)

        assert config.task in ["subtype_classification", "survival"], f"Unknown task '{config.task}'."
        assert config.magnification_factor in [2, 4], f"Only M=2 and M=4 supported."

        if config.multi_dataset is not None:
            assert config.task == "subtype_classification", "multi_dataset only supported for subtype classification"

        return config

    def power_levels(self):
        return [self.base_power * self.magnification_factor ** i for i in range(self.num_levels)]

    def get_model(self):
        if self.model_type == "PATHS":
            return RecursiveModel(PATHSProcessor, self.model_config, train_config=self)
        elif self.model_type == "abmil":
            return ABMIL(self.model_config, self)
        elif self.model_type == "transmil":
            return TransMIL(self.model_config, self)
        elif self.model_type.lower() == "ilra":
            return ILRA(self.model_config, self)
        else:
            raise NotImplementedError(f"Unknown model '{self.model_type}'.")

    # Load train/test/val split with proportions given by props (a list of 3 floats)
    def get_dataset(self, props, seed, ctx_dim, **kwargs):
        return load_splits(props, seed, ctx_dim, self, **kwargs)

    def get_lr_scheduler(self, optimizer):
        return ExponentialLR(optimizer, self.lr_decay_per_epoch)

    def num_logits(self) -> int:
        if self.task == "survival":
            return self.nbins
        elif self.filter_to_subtypes != None:
            return len(self.filter_to_subtypes)
        else:
            return len(self.multi_dataset)
