import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from text_autocompl.files import read_config
from text_autocompl.log import get_logger


def get_dataset(config_path="./config.yaml", logger=None):
    if logger is None:
        logger = get_logger()

    config = read_config(config_path, logger)
    cache_dir = Path(config["data_dir"]).joinpath("raw")
    cache_dir.mkdir(exist_ok=True, parents=True)

    logger.debug("Load from 'wikitext-2-raw-v1' form 'Salesforce/wikitext'")

    return load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        cache_dir=str(cache_dir),
    )


def dataset_info(hf_dataset, n_random_examples=5, bins=17):
    dataset_len = len(hf_dataset)
    print(f"Длина датасета: {dataset_len}\n")

    selected_ids = np.random.permutation(dataset_len)[
        :n_random_examples
    ].tolist()

    print("Примеры текстов:\n")
    for text_id in selected_ids:
        print(f"id={text_id}")
        print(hf_dataset[text_id]["text"])
        print()

    lengths = np.array([len(text.split()) for text in hf_dataset["text"]])

    print(f"Минимальная длина: {lengths.min()}")
    print(f"Медианная длина: {np.median(lengths)}")
    print(f"Средняя длина: {lengths.mean()}")
    print(f"Максимальная длина: {lengths.max()}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(lengths, bins=bins)
    ax[0].set_title("Распределение длин текстов")

    ax[1].hist(lengths[lengths > 0], bins=bins)
    ax[1].set_title("Распределение длин\nтекстов с длиной больше 0")

    for axes in ax.reshape(-1):
        axes.set_xlabel("Длина текста")
        axes.set_ylabel("Количество текстов")

    plt.tight_layout()
    plt.show()


class WordTokenizer:
    def __init__(self, config_path="./config.yaml", logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = get_logger()

        self.logger.debug(f"Init WordTokenizer config_path='{config_path}'")

        config = read_config(config_path, logger)
        self.data_dir = config["data_dir"]
        self.n_most_freq_words = config["tokenizer"]["n_most_freq_words"]

        vocab_file = Path(self.data_dir).joinpath("vocab.json")
        if not vocab_file.exists():
            self.logger.debug(f"{vocab_file} doesn't exist. Create a new one.")
            texts = get_dataset(config_path=config_path, logger=self.logger)[
                "train"
            ]

            self.vocab = Counter({"<PAD>": float("inf"), "<UNK>": float("inf")})
            for text in texts:
                self.vocab.update(text["text"].split())

            self.vocab = {
                key: {"id": i, "freq": freq}
                for i, (key, freq) in enumerate(self.vocab.most_common())
            }

            self.logger.debug(f"New vocabulary created. Save to {vocab_file}.")
            with open(vocab_file, "w") as f:
                json.dump(self.vocab, f, indent=4)
        else:
            self.logger.debug(f"Load vocabulary from {vocab_file}.")
            with open(vocab_file, "r") as f:
                self.vocab = json.load(f)

        if self.n_most_freq_words:
            self.vocab = {
                key: val
                for key, val in self.vocab.items()
                if val["id"] < self.n_most_freq_words
            }

        self.rev_vocab = {val["id"]: key for key, val in self.vocab.items()}

    def encode(self, text):
        tokens = text.split()
        unk_item = self.vocab["<UNK>"]
        ids = [self.vocab.get(token, unk_item)["id"] for token in tokens]
        return {
            "input_ids": ids,
            "tokens": tokens,
        }

    def decode(self, input_ids):
        return [self.rev_vocab[token_id] for token_id in input_ids]

    @property
    def pad_token_id(self):
        return self.vocab["<PAD>"]["id"]

    @property
    def unk_token_id(self):
        return self.vocab["<UNK>"]["id"]

    def __call__(self, text):
        return self.encode(text)


class WikiDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        config_path="./config.yaml",
        logger=None,
    ):
        self.logger = logger
        if self.logger is None:
            self.logger = get_logger()

        self.logger.debug(
            f"Init WikiDataset split='{split}', "
            f"tokenizer='{tokenizer}', "
            f"config_path='{config_path}'"
        )

        config = read_config(config_path, logger)

        self.data = get_dataset(config_path=config_path, logger=self.logger)[
            split
        ]["text"]

        self.tokenizer = tokenizer
        self.data = [self.tokenizer.encode(text) for text in self.data]

        self.max_len = config["tokenizer"]["max_len"]

        # Если max_len не None, чтобы не потерять данные будем нарезать тексты
        # на куски длиной не больше max_len. Индексы начала этих кусков будем
        # хранить в self.indices ([[text_id, chunk_start_id_in_text]]).
        # Одновременно будем отфильтровывать тексты длиной меньше двух токенов.
        self.indices = list()
        for text_idx, item in enumerate(self.data):
            text = item["input_ids"]
            n_tokens = len(text)
            if n_tokens < 2:
                continue

            if self.max_len is None:
                # текст целиком является одним chunk-ом
                self.indices.append((text_idx, 0))
            else:
                # текст можно разделить на несколько chunk-ов
                for chunk_idx in range(0, n_tokens, self.max_len):
                    # теоретически последний чанк в тексте может состоять из
                    # одного токена, поэтому добавляем чанк только если его
                    # длина больше 2.
                    if n_tokens - chunk_idx > 1:
                        self.indices.append((text_idx, chunk_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        text_idx, chunk_idx = self.indices[idx]

        tokens = self.data[text_idx]["tokens"]
        input_ids = self.data[text_idx]["input_ids"]

        if self.max_len is not None:
            tokens = tokens[chunk_idx : chunk_idx + self.max_len + 1]
            input_ids = input_ids[chunk_idx : chunk_idx + self.max_len + 1]

        return {
            "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(input_ids[1:], dtype=torch.long),
            "tokens": tokens[:-1],
            "target_tokens": tokens[1:],
        }


def data_collator(batch, pad_token_id=0, max_len=None):
    sorted_batch = sorted(
        batch, key=lambda x: len(x["input_ids"]), reverse=True
    )

    if max_len is None:
        input_ids = [item["input_ids"] for item in sorted_batch]
        labels = [item["labels"] for item in sorted_batch]
    else:
        input_ids = [item["input_ids"][:max_len] for item in sorted_batch]
        labels = [item["labels"][:max_len] for item in sorted_batch]

    input_lengths = torch.tensor([len(item) for item in input_ids])
    label_lengths = torch.tensor([len(item) for item in labels])

    padded_input = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    ).long()
    padded_labels = pad_sequence(
        labels, batch_first=True, padding_value=pad_token_id
    ).long()

    mask_input = (padded_input != pad_token_id).long()
    mask_labels = (padded_labels != pad_token_id).long()

    return {
        "input_ids": padded_input,
        "labels": padded_labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
        "mask_input": mask_input,
        "mask_labels": mask_labels,
    }
