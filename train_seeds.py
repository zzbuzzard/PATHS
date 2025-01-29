import numpy as np
import torch
import argparse
import wandb
import os
from dataclasses import asdict

import utils
import config as cfg
from utils import device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory (seed 0). Must contain "
                                                                 "config.json file.")
    parser.add_argument("--wandb-project-name", type=str, default="hierarchical-cv-2")
    args = parser.parse_args()

    root = args.model_dir
    config = cfg.Config.load(root + "_0")

    if config.task == "survival":
        num_seeds = 5
    else:
        num_seeds = 10

    if config.model_type.upper() == "PATHS":
        from train import train_loop
    else:
        from train_baseline import train_loop

    print("Running for", num_seeds, "seeds")

    for seed in range(num_seeds):
        args.model_dir = root + "_" + str(seed)
        print("RUNNNING SEED", seed, ":", args.model_dir)

        config = cfg.Config.load(args.model_dir)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        model = config.get_model()
        model = model.train().to(device)

        name = os.path.split(args.model_dir)[-1]
        run_id = utils.wandb_get_id(args.model_dir)

        wandb.init(
            project=args.wandb_project_name,
            name=f"{name}",
            config=asdict(config),
            resume="allow",
            id=run_id,
            reinit=True
        )

        wandb.define_metric("epoch")

        if config.model_type.upper() == "PATHS":
            train, val, test = config.get_dataset([0.7, 0.15, 0.15], config.seed, model.procs[0].ctx_dim())
        else:
            train, val, test = config.get_dataset([0.7, 0.15, 0.15], config.seed, (0, 0))

        if config.early_stopping:
            assert val is not None, f"Must have validation set to use early stopping"

        print("VAL IS", val)
        train_loop(model, train, val, test, config, args.model_dir)

    wandb.finish()
