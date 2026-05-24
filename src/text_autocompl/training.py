from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from text_autocompl.data import WikiDataset, WordTokenizer, data_collator
from text_autocompl.files import read_config
from text_autocompl.log import get_logger
from text_autocompl.models import RecNN


class Perplexity:
    def __init__(self, batch=False, **cross_entropy_kwargs):
        self.cross_entropy = torch.nn.CrossEntropyLoss(**cross_entropy_kwargs)
        self.batch = batch

    def __call__(self, logits, labels):
        loss = self.cross_entropy(logits, labels)
        if self.batch:
            loss *= len(logits)

        return torch.exp(loss)


def load_additional_metrics(config_path="./config.yaml", logger=None):
    if logger is None:
        logger = get_logger()

    config = read_config(config_path, logger)
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


def save_checkpoint(path, model, optimizer, epoch, **kwargs):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)


def train(config_path="./config.yaml", logger=None):
    if logger is None:
        logger = get_logger()

    logger.info("Start training")

    parameters = read_config(config_path, logger)
    logger.debug(f"Loaded from {config_path} next parameters: {parameters}")

    vocab_size = parameters["tokenizer"]["vocab_size"]
    models_dir = Path(parameters["models_dir"])
    models_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = WordTokenizer(config_path=config_path, logger=logger)
    logger.info(f"Initialized the tokenizer")

    pad_token_id = tokenizer.pad_token_id
    logger.debug(f"Pad token id: {pad_token_id}")

    train_dataset = WikiDataset(
        tokenizer=tokenizer,
        split="train",
        config_path=config_path,
        logger=logger,
    )
    logger.info("Initialize train dataset")

    val_dataset = WikiDataset(
        tokenizer=tokenizer,
        split="validation",
        config_path=config_path,
        logger=logger,
    )
    logger.info("Initialize validation dataset")

    collate_fn = lambda batch: data_collator(
        batch,
        pad_token_id=pad_token_id,
        max_len=parameters["tokenizer"]["max_len"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=parameters["training_params"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=parameters["training_params"]["num_workers"],
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=parameters["training_params"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=parameters["training_params"]["num_workers"],
    )

    logger.info("Initialize dataloaders")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    model_type = parameters["model"]["model_type"]
    if model_type not in ["LSTM", "GRU"]:
        raise ValueError("Support only LSTM and GRU")

    model = RecNN(
        cell_type=model_type,
        vocab_size=vocab_size,
        pad_idx=pad_token_id,
        **parameters["model"]["model_params"],
    )
    logger.info(f"Initialized the model. Model type: {model_type}")

    if parameters["model"]["checkpoint_name"] is not None:
        checkpoint_path = models_dir.joinpath(
            parameters["model"]["checkpoint_name"]
        )
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warinig(
                f"Checkpoint {checkpoint_path} doesn't exist. "
                "Training from scratch"
            )
            checkpoint = None
    else:
        logger.info("Training from scratch")
        checkpoint = None

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model weights from {checkpoint_path}")

    model.to(device)
    logger.debug(f"Moved the model to {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **parameters["optimizer_params"],
    )
    logger.info("Initialized the optimizer")

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded optimizer state from {checkpoint_path}")

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=parameters["training_params"]["label_smoothing"],
        ignore_index=pad_token_id,
    )
    logger.info("Initialized the loss function")

    n_epochs = parameters["training_params"]["n_epochs"]

    start_training_time = datetime.now()
    metrics_dir = Path(parameters["metrics_dir"])
    metrics_dir.mkdir(exist_ok=True, parents=True)
    metrics_f_path = metrics_dir.joinpath(
        f"metrics_{start_training_time.strftime('%Y_%m_%dT%H_%M_%S.csv')}"
    )
    metrics_df = pd.DataFrame(
        columns=[
            "time",
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_EM",
            "val_F1",
            "val_CER",
            "val_preplexity",
        ]
    )

    logger.info(f"Start training loop. N epocs = {n_epochs}")
    with logging_redirect_tqdm():
        for epoch in tqdm(range(n_epochs)):
            logger.info(f"Epoch {epoch} / {n_epochs}. Training")

            model.train()

            train_loss = 0
            train_acc = 0
            n_samples = 0

            for batch in tqdm(train_dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                lengths = batch["input_lengths"]

                logits = model(input_ids, lengths)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    **parameters["grad_clipping_params"],
                )
                optimizer.step()

                n_samples += len(logits)
                train_loss = loss.detach().cpu().item() * len(logits)
                with torch.no_grad():
                    train_acc += get_accuracy(
                        logits.cpu(),
                        labels.cpu(),
                        batch_agg="sum",
                    ).item()
            train_loss /= n_samples
            train_acc /= n_samples

            logger.info(f"Epoch {epoch} / {n_epochs}. Validation")
            eval_metrics_dict = evaluate(
                dataloader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                device=device,
                tokenizer=tokenizer,
            )

            epoch_finish_time = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
            new_metrics = {
                "time": epoch_finish_time,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": eval_metrics_dict["loss"],
                "val_acc": eval_metrics_dict["acc"],
                "val_EM": eval_metrics_dict["exact_match"],
                "val_F1": eval_metrics_dict["f1"],
                "val_CER": eval_metrics_dict["cer"],
                "val_preplexity": eval_metrics_dict["perplexity"],
            }

            logger.info(f"Epoch {epoch} / {n_epochs}. Metrics: {new_metrics}")

            metrics_df = pd.concat(
                [metrics_df, pd.DataFrame(new_metrics)],
                ignore_index=True,
            )
            metrics_df.to_csv(metrics_f_path, index=False)

            logger.info(f"Save metrics to {metrics_f_path}")

            save_checkpoint(
                models_dir.joinpath(
                    f"{model_type}_epoch_{epoch:04d}_{epoch_finish_time}.pt"
                ),
                model,
                optimizer,
                epoch,
                train_loss=train_loss,
                val_loss=eval_metrics_dict["loss"],
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    return metrics_df, model


def evaluate(
    dataloader,
    model,
    loss_fn,
    device,
    tokenizer,
    parameters,
    config_path="./config.yaml",
    logger=None,
):

    model.eval()
    n_samples = 0
    loss = 0
    acc = 0
    perplexity = 0

    perplexity = Perplexity(
        batch=False,
        label_smoothing=parameters["training_params"]["label_smoothing"],
        ignore_index=tokenizer.pad_token_id,
    )

    additional_metrics = load_additional_metrics(
        config_path=config_path, logger=logger
    )
    exact_match = additional_metrics["exact_match"]
    f1 = additional_metrics["f1"]
    cer = additional_metrics["cer"]

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["input_lengths"]

            logits = model(input_ids, lengths)

            n_samples += len(logits)
            loss += loss_fn(logits, labels).cpu().item() * len(logits)
            perplexity += perplexity(logits, labels).cpu().item()

            logits = logits.cpu()
            labels = labels.cpu()

            acc += get_accuracy(logits, labels, batch_agg="sum").item()

            preds = logits.argmax(dim=-1)
            f1.add_batch(predictions=preds, references=labels)

            preds_text = [tokenizer.decode(row) for row in preds.tolist()]
            labels_text = [tokenizer.decode(row) for row in labels.tolist()]
            exact_match.add_batch(
                predictions=preds_text, references=labels_text
            )
            cer.add_batch(predictions=preds_text, references=labels_text)

        loss /= n_samples
        acc /= n_samples
        perplexity /= n_samples
        f1 = f1.compute(average="macro")
        exact_match = exact_match.compute()
        cer = cer.compute()

    return dict(
        loss=loss,
        acc=acc,
        perplexity=perplexity,
        f1=f1,
        exact_match=exact_match,
        cer=cer,
    )


def test():
    pass


def predict():
    pass
