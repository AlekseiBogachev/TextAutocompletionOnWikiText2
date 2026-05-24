import logging
from datetime import datetime
from pathlib import Path


def get_logger(logs_dir="./logs", log_level=logging.INFO):
    logs_dir = Path(logs_dir)

    txt_logger = logging.getLogger("txt_logger")
    txt_logger.setLevel(log_level)
    log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(
        logs_dir.joinpath(datetime.now().strftime("%Y_%m_%dT%H_%M_%S.log")),
        mode="w",
    )
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    fh.setFormatter(log_formatter)
    ch.setFormatter(log_formatter)
    txt_logger.addHandler(fh)
    txt_logger.addHandler(ch)

    return txt_logger
