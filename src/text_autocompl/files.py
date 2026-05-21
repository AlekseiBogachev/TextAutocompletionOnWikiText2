from pathlib import Path
import yaml

from text_autocompl import get_logger

def read_config(path="./config.yaml", logger=None):
    if logger is None:
        logger = get_logger()

    config_path = Path(path)
    logger.debug(f"Read config from the file '{path}'")

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as exception:
        logger.error(exception)
    except yaml.YAMLError as exception:
        logger.error(exception)
