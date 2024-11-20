import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from abc import ABC, abstractmethod
from typing import Dict
from torcheval.metrics import BinaryAUROC


class Evaluator(ABC):
    """Keeps track of various statistics e.g. mean loss, accuracy or c-index."""
    def __init__(self, split: str):
        self.losses = []
        self.split = split

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def register(self, *args):
        raise NotImplementedError

    @abstractmethod
    def calculate(self, train_stats=None, epoch=None) -> Dict:
        raise NotImplementedError

    def _add_to_train_stats(self, epoch, out, train_stats):
        if train_stats is not None:
            # Update train_stats
            for key in out:
                if key in train_stats:
                    if epoch is None:
                        train_stats[key] = out[key]
                    else:
                        train_stats[key][epoch] = out[key]


class SurvivalEvaluator(Evaluator):
    def __init__(self, split: str):
        super().__init__(split)

        # For c-index
        self.all_censorships = []
        self.all_event_times = []
        self.all_risk_scores = []

    def reset(self):
        self.losses.clear()
        self.all_censorships.clear()
        self.all_event_times.clear()
        self.all_risk_scores.clear()

    def register(self, batch, hazards, loss):
        censors = batch["censored"]

        # Track loss
        self.losses.append(loss.item())

        # Track stats for c-index
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)
        self.all_censorships.append(censors.detach().cpu().numpy())
        self.all_event_times.append(batch["survival"].detach().cpu().numpy())
        self.all_risk_scores.append(risk.detach().cpu().numpy())

    def calculate(self, train_stats=None, epoch=None):
        # censors=1 -> censored
        # censors=0 -> didnt occur
        all_censorships = (1 - np.concatenate(self.all_censorships)).astype(np.bool_)
        all_event_times = np.concatenate(self.all_event_times)
        all_risk_scores = np.concatenate(self.all_risk_scores)

        if np.sum(all_censorships).item() <= 1:
            print("Warning: all events censored")
            c_index = 0.5
        else:
            c_index = concordance_index_censored(all_censorships, all_event_times, all_risk_scores)[0]

        out = {
            f"{self.split}_loss": sum(self.losses) / len(self.losses),
            f"{self.split}_c-index": c_index
        }
        self._add_to_train_stats(epoch, out, train_stats)
        return out


class SubtypeClassificationEvaluator(Evaluator):
    def __init__(self, split: str, nclasses: int):
        super().__init__(split)

        self.nclasses = nclasses
        self.metrics = [BinaryAUROC() for _ in range(nclasses)]

    def reset(self):
        self.losses.clear()
        for m in self.metrics:
            m.reset()

    def register(self, batch, logits, loss):
        # Track loss
        self.losses.append(loss.item())

        subtypes = batch["subtype"].detach().cpu()
        preds = torch.softmax(logits, dim=-1).detach().cpu()

        for i in range(self.nclasses):
            i_preds = preds[:, i]
            i_subtype = (subtypes == i).to(torch.long)  # 1 where subtypes=i else 0
            self.metrics[i].update(i_preds, i_subtype)

    def calculate(self, train_stats=None, epoch=None):
        aucs = [m.compute().item() for m in self.metrics]
        mean_auc = sum(aucs) / len(aucs)

        out = {
            f"{self.split}_loss": sum(self.losses) / len(self.losses),
            f"{self.split}_AUC": mean_auc
        }
        self._add_to_train_stats(epoch, out, train_stats)
        return out



