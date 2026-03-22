import os
import logging
from .ddp_utils import is_main_process


def logger_configuration(config, save_log=False, test_mode=False):
    logger = logging.getLogger("Deep joint source channel coder")
    # Avoid re-adding handlers in case of multiple imports / repeated calls
    if logger.hasHandlers():
        logger.handlers.clear()
    if test_mode:
        config.workdir += "_test"
    if save_log and is_main_process():
        os.makedirs(config.workdir, exist_ok=True)
        os.makedirs(config.samples, exist_ok=True)
        os.makedirs(config.models, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s] %(message)s")
    # --- Only rank 0 prints to stdout ---
    if is_main_process():
        stdhandler = logging.StreamHandler()
        stdhandler.setLevel(logging.INFO)
        stdhandler.setFormatter(formatter)
        logger.addHandler(stdhandler)
    # --- All ranks can log to file if you want shared logs ---
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def get_logger_dir(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return os.path.dirname(handler.baseFilename)
    return "."
