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
from eval import Evaluator, SurvivalEvaluator, SubtypeClassificationEvaluator
from train import get_dataloaders


def train_loop(model: RecursiveModel, train_ds: SlideDataset, val_ds: SlideDataset, test_ds: SlideDataset, config: cfg.Config, model_dir: str):
    def mk_eval(split: str):
        if config.task == "subtype_classification":
            return SubtypeClassificationEvaluator(split, config.num_logits())
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

        for idx, batch in enumerate(tqdm(train_loader)):
            with autocast(enabled=config.use_mixed_precision):
                hazards_or_logits, loss = utils.inference_baseline(model, batch, config.task, config.model_type)

            # Handle gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            call_optimize = ((idx + 1) % config.gradient_accumulation_steps == 0) or (idx == len(train_loader) - 1)

            if config.use_mixed_precision:
                scaler.scale(loss).backward()

                if call_optimize:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
            else:
                loss.backward()

                if call_optimize:
                    opt.step()
                    opt.zero_grad()

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
                    hazards_or_logits, loss = utils.inference_baseline(model, batch, config.task, config.model_type)

                    val_eval.register(batch, hazards_or_logits, loss)

                log_dict = val_eval.calculate(train_stats, e) | {"epoch": e}
                wandb.log(log_dict)
                val_eval.reset()

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
            hazards_or_logits, loss = utils.inference_baseline(model, batch, config.task, config.model_type)

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

    train, val, test = config.get_dataset([0.7, 0.15, 0.15], config.seed, (0, 0))
    if config.early_stopping:
        assert val is not None, f"Must have validation set to use early stopping"

    print("VAL IS", val)
    train_loop(model, train, val, test, config, args.model_dir)

    wandb.finish()

