from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from text_autocompl.data import (
    WikiDataset,
    WordTokenizer,
    data_collator,
    get_dataset,
)
from text_autocompl.log import get_logger
from text_autocompl.metrics import get_accuracy
from text_autocompl.models import RecNN


def save_checkpoint(path, model, optimizer, epoch, **kwargs):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)


def load_checkpoint(config, logger=None):
    if logger is None:
        logger = get_logger()

    models_dir = Path(config["models_dir"])
    if config["model"]["checkpoint_name"] is not None:
        checkpoint_path = models_dir.joinpath(
            config["model"]["checkpoint_name"]
        )
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warinig(
                f"Checkpoint {checkpoint_path} doesn't exist. "
                "Init model from scratch"
            )
            checkpoint = None
    else:
        logger.info("Init model from scratch")
        checkpoint = None

    return checkpoint, checkpoint_path


def init_custom_model(config, pad_token_id, logger):
    model_type = config["model"]["model_type"]
    if model_type not in ["LSTM", "GRU"]:
        raise ValueError("Support only LSTM and GRU")

    model = RecNN(
        cell_type=model_type,
        vocab_size=config["tokenizer"]["vocab_size"],
        pad_idx=pad_token_id,
        **config["model"]["model_params"],
    )
    logger.info(f"Initialized the model. Model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    return model


def train(config, logger=None):
    if logger is None:
        logger = get_logger()

    logger.info("Start training")

    logger.debug(f"Parameters: {config}")

    models_dir = Path(config["models_dir"])
    models_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = WordTokenizer(config=config, logger=logger)
    logger.info(f"Initialized the tokenizer")

    pad_token_id = tokenizer.pad_token_id
    logger.debug(f"Pad token id: {pad_token_id}")

    train_dataset = WikiDataset(
        tokenizer=tokenizer,
        split="train",
        config=config,
        logger=logger,
    )
    logger.info("Initialize train dataset")

    val_dataset = WikiDataset(
        tokenizer=tokenizer,
        split="validation",
        config=config,
        logger=logger,
    )
    logger.info("Initialize validation dataset")

    collate_fn = lambda batch: data_collator(
        batch,
        pad_token_id=pad_token_id,
        max_len=config["tokenizer"]["max_len"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["training_params"]["num_workers"],
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["training_params"]["num_workers"],
    )

    logger.info("Initialize dataloaders")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    model = init_custom_model(config, pad_token_id, logger)
    checkpoint, checkpoint_path = load_checkpoint(config, logger)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model weights from {checkpoint_path}")

    model.to(device)
    logger.debug(f"Moved the model to {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config["optimizer_params"],
    )
    logger.info("Initialized the optimizer")

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded optimizer state from {checkpoint_path}")

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["training_params"]["label_smoothing"],
        ignore_index=pad_token_id,
    )
    logger.info("Initialized the loss function")

    n_epochs = config["training_params"]["n_epochs"]

    start_training_time = datetime.now()
    metrics_dir = Path(config["metrics_dir"])
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
            "train_perplexity",
            "val_loss",
            "val_acc",
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
                    **config["grad_clipping_params"],
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
            train_perplexity = np.exp(train_loss).item()
            train_acc /= n_samples

            logger.info(f"Epoch {epoch} / {n_epochs}. Validation")
            eval_metrics_dict = evaluate(
                dataloader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                device=device,
            )

            epoch_finish_time = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
            new_metrics = {
                "time": epoch_finish_time,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "train_acc": train_acc,
                "val_loss": eval_metrics_dict["loss"],
                "val_acc": eval_metrics_dict["acc"],
                "val_preplexity": eval_metrics_dict["perplexity"],
            }

            logger.info(f"Epoch {epoch} / {n_epochs}. Metrics: {new_metrics}")

            metrics_df = pd.concat(
                [metrics_df, pd.DataFrame(new_metrics)],
                ignore_index=True,
            )
            metrics_df.to_csv(metrics_f_path, index=False)

            logger.info(f"Save metrics to {metrics_f_path}")

            model_type = config["model"]["model_type"]
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
):

    model.eval()
    n_samples = 0
    loss = 0
    acc = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["input_lengths"]

            logits = model(input_ids, lengths)

            n_samples += len(logits)
            loss += loss_fn(logits, labels).cpu().item() * len(logits)

            logits = logits.cpu()
            labels = labels.cpu()

            acc += get_accuracy(logits, labels, batch_agg="sum").item()

        loss /= n_samples
        acc /= n_samples
        perplexity = np.exp(loss).item()

    return dict(
        loss=loss,
        acc=acc,
        perplexity=perplexity,
    )


def test_custom_model(config, logger=None):
    if logger is None:
        logger = get_logger()

    logger.info("Start custom model test")

    logger.debug(f"Parameters: {config}")

    models_dir = Path(config["models_dir"])
    models_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = WordTokenizer(config=config, logger=logger)
    logger.info(f"Initialized the tokenizer")

    pad_token_id = tokenizer.pad_token_id
    logger.debug(f"Pad token id: {pad_token_id}")

    test_dataset = WikiDataset(
        tokenizer=tokenizer,
        split="test",
        config=config,
        logger=logger,
    )
    logger.info("Initialize test dataset")

    collate_fn = lambda batch: data_collator(
        batch,
        pad_token_id=pad_token_id,
        max_len=config["tokenizer"]["max_len"],
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["training_params"]["num_workers"],
    )
    logger.info("Initialize test dataloader")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    model = init_custom_model(config, pad_token_id, logger)
    checkpoint, checkpoint_path = load_checkpoint(config, logger)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model weights from {checkpoint_path}")

    model.to(device)
    logger.debug(f"Moved the model to {device}")

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config["training_params"]["label_smoothing"],
        ignore_index=pad_token_id,
    )

    logger.info("Evaluate model on test set")
    test_metrics_dict = evaluate(
        dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        device=device,
    )
    logger.info(f"Test metrics: {test_metrics_dict}")

    test_metrics_df = pd.DataFrame.from_dict(test_metrics_dict)

    model_type = config["model"]["model_type"]
    metrics_dir = Path(config["metrics_dir"])
    metrics_dir.mkdir(exist_ok=True, parents=True)
    metrics_f_path = metrics_dir.joinpath(
        f"test_metrics_{model_type}_"
        f"{datetime.now().strftime('%Y_%m_%dT%H_%M_%S.csv')}"
    )
    test_metrics_df.to_csv(metrics_f_path, index=False)

    logger.info(f"Save metrics to {metrics_f_path}")

    return test_metrics_df


def test_distilgpt2(config, logger=None):
    if logger is None:
        logger = get_logger()

    model_name = "distilbert/distilgpt2"
    logger.info(f"Тестирование {model_name}")
    logger.debug(f"Parameters: {config}")

    hf_dataset = get_dataset(config, logger=None)["test"]
    logger.info("Loaded test dataset")

    cache_dir = str(Path(config["models_dir"]).joinpath("distilgpt2_tokenizer"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    logger.info(f"Loaded tokenizer to {cache_dir}")

    cache_dir = str(Path(config["models_dir"]).joinpath("distilgpt2_model"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    logger.info(f"Loaded model to {cache_dir}")

    # токенизатор для distilgpt2 не имеет своего pad_token, мы должны
    # его назанчить
    tokenizer.pad_token = tokenizer.eos_token

    def tokenization(example):
        return tokenizer(
            example["text"],
            return_tensors="pt",
            max_length=config["distilgpt2"]["max_len"],
            truncation=True,
        )

    min_text_len = config["distilgpt2"]["min_text_len"]
    hf_dataset = hf_dataset.filter(lambda x: len(x["text"]) > min_text_len)
    logger.info(f"Deleted text with length less than {min_text_len}")

    tokenized_dataset = hf_dataset.map(
        tokenization,
        batched=True,
        remove_columns=["text"],  # исходные тексты больше не нужны
    )
    logger.info("Tokenized dataset")

    # Автоматически создаст 'labels' из 'input_ids'
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # предсказываем следующий токет - Causal LM
    )
    logger.info("Created data_collator")

    test_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=config["distilgpt2"]["batch_size"],
        shuffle=False,
        collate_fn=data_collator,
    )
    logger.info("Created test DataLoader")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    model.to(device)
    logger.debug(f"Moved the model to {device}")

    model.eval()

    logger.info("Evaluate model on test set")
    test_loss = 0.0
    test_acc = 0.0
    n_samples = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # batch = {input_ids: ..., attention_mask: ..., labels: ...}
            # сдвиг labels на 1 токен вперёд относительно input_ids
            # происходит внутри самой модели
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)

            labels = batch["labels"]
            logits = outputs.logits
            loss = outputs.loss.item() * batch_len

            batch_len = len(logits)
            test_loss += loss
            n_samples += batch_len

            shifted_preds = logits[:, :-1, :].argmax(dim=-1)
            shifted_labels = labels[:, 1:]
            # DataCollatorForLanguageModeling заменяет tokenizer.pad_token_id
            # на -100 в таргете, чтобы не учитывать паддинг при расчёте loss.
            mask = shifted_labels != -100

            test_acc += ((shifted_preds == shifted_labels) * mask).sum().item()
            total_tokens += mask.sum().item()

        loss /= n_samples
        test_acc /= total_tokens
        perplexity = np.exp(loss).item()

    test_metrics_dict = dict(
        loss=loss,
        acc=test_acc,
        perplexity=perplexity,
    )

    logger.info(f"Test metrics: {test_metrics_dict}")

    test_metrics_df = pd.DataFrame.from_dict(test_metrics_dict)

    metrics_dir = Path(config["metrics_dir"])
    metrics_dir.mkdir(exist_ok=True, parents=True)
    metrics_f_path = metrics_dir.joinpath(
        "test_metrics_distilgpt2_"
        f"{datetime.now().strftime('%Y_%m_%dT%H_%M_%S.csv')}"
    )
    test_metrics_df.to_csv(metrics_f_path, index=False)
    logger.info(f"Save metrics to {metrics_f_path}")

    return test_metrics_df


def predict(text, config, logger=None):
    if logger is None:
        logger = get_logger()

    raise NotImplementedError
