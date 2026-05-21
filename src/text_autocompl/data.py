from pathlib import Path

from datasets import load_dataset

from text_autocompl import read_config


def get_dataset(config_path="./config.yaml", logger=None):
    config = read_config(config_path, logger)
    cache_dir = Path(config["data_dir"])
    cache_dir.mkdir(exist_ok=True, parents=True)

    return load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        cache_dir=str(cache_dir),
    )
