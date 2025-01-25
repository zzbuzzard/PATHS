import numpy as np
import torch
import torch.utils.data as dutils
import argparse
import wandb
from tqdm import tqdm
import os
from dataclasses import asdict
from torch.cuda.amp import GradScaler, autocast

import utils
import config as cfg
from utils import device, EarlyStopping
from data_utils.dataset import SlideDataset, collate_fn
from model.interface import RecursiveModel
from eval import SurvivalEvaluator, SubtypeClassificationEvaluator


def get_dataloaders(train, val, test, batch_size):
    num_workers = 0
    prefetch = None

    train_dataloader = dutils.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                         num_workers=num_workers, prefetch_factor=prefetch)
    val_dataloader = dutils.DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if val is not None else None
    test_dataloader = dutils.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader


def train_loop(model: RecursiveModel, train_ds: SlideDataset, val_ds: SlideDataset, test_ds: SlideDataset, config: cfg.Config, model_dir: str):
    def mk_eval(split: str):
        if config.task == "subtype_classification":
            return SubtypeClassificationEvaluator(split, len(config.filter_to_subtypes))
        else:
            return SurvivalEvaluator(split)

    train_stats = utils.load_state(model_dir, model)

    start_epoch = train_stats["epoch"]
    for key in ["train_loss", "train_c-index", "val_loss", "val_c-index"]:
        if key not in train_stats:
            train_stats[key] = {}

    print("Training starts at epoch", start_epoch)

    train_eval, val_eval = mk_eval("train"), mk_eval("val")

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = config.get_lr_scheduler(opt)

    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, config.batch_size[0])

    scaler = GradScaler()

    # For early stopping on val loss
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="min", verbose=True)
    if config.early_stopping:
        assert val_loader is not None, f"A validation set must be used when early stopping is enabled."

    for e in range(start_epoch, config.num_epochs + 1):
        print("Epoch", e, "/", config.num_epochs)

        for batch in tqdm(train_loader):
            opt.zero_grad()

            hazards_or_logits, loss = utils.inference_end2end(config.num_levels, config.top_k_patches, model,
                                                              config.base_power, batch, config.task,
                                                              config.use_mixed_precision)

            if config.use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            train_eval.register(batch, hazards_or_logits, loss)

        print("num_ims:", batch["num_ims"])

        lr_scheduler.step()
        wandb.log(train_eval.calculate(train_stats, e) | {"epoch": e})
        train_eval.reset()

        train_stats["epoch"] = e + 1

        if e % config.eval_epochs == 0 and val_loader is not None:
            print("Evaluating on validation set...")
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    hazards_or_logits, loss = utils.inference_end2end(config.num_levels, config.top_k_patches, model,
                                                                      config.base_power, batch, config.task)

                    val_eval.register(batch, hazards_or_logits, loss)

                log_dict = val_eval.calculate(train_stats, e) | {"epoch": e}
                wandb.log(log_dict)
                val_eval.reset()

                # val_score = log_dict["val_c-index"] if config.task == "survival" else log_dict["val_AUC"]
                val_score = log_dict["val_loss"]
                if config.early_stopping and early_stopping.step(val_score, model):
                    print(f"Early stopping at epoch {e+1}")
                    model = early_stopping.load_best_weights(model)
                    break

            model.train()

    utils.save_state(model_dir, model, train_stats)

    # Training has completed, evaluate on test
    model.eval()
    test_eval = mk_eval("test")
    with torch.no_grad():
        for batch in test_loader:
            hazards_or_logits, loss = utils.inference_end2end(config.num_levels, config.top_k_patches, model,
                                                              config.base_power, batch, config.task)

            test_eval.register(batch, hazards_or_logits, loss)

    wandb.log(test_eval.calculate(train_stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory. Must contain "
                                                                 "config.json file.")
    parser.add_argument("--wandb-project-name", type=str, default="PATHS")
    args = parser.parse_args()

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
        id=run_id
    )

    wandb.define_metric("epoch")
    for split in ["train", "test", "val"]:
        wandb.define_metric(f"{split}_loss", step_metric="epoch")
        wandb.define_metric(f"{split}_accuracy", step_metric="epoch")
        wandb.define_metric(f"{split}_c-index", step_metric="epoch")

    train, val, test = config.get_dataset([0.7, 0.15, 0.15], config.seed, model.procs[0].ctx_dim())
    if config.early_stopping:
        assert val is not None, f"Must have validation set to use early stopping"

    print("VAL IS", val)
    train_loop(model, train, val, test, config, args.model_dir)

    wandb.finish()
