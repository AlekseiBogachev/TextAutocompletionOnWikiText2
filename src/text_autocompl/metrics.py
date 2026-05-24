from pathlib import Path

import evaluate
import torch

from text_autocompl.log import get_logger


def load_additional_metrics(config, logger=None):
    if logger is None:
        logger = get_logger()

    metrics_cache_dir = Path(config["metrics_dir"])
    metrics_names = [
        "exact_match",
        "f1",
        "cer",
    ]

    metrics_fns = dict()
    for name in metrics_names:
        metrics_fns[name] = evaluate.load(
            name, cache_dir=metrics_cache_dir.joinpath(name)
        )
        logger.debug(f"Loaded '{name}' metric")

    return metrics_fns


def get_accuracy(logits, labels, batch_agg="mean"):
    preds = torch.argmax(logits, dim=-1)
    if batch_agg == "sum":
        return (preds == labels).sum()
    elif batch_agg == "mean":
        return (preds == labels).mean()
    else:
        return preds == labels
