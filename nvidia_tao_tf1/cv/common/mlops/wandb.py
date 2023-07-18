# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

"""Routines for connecting with Weights and Biases client."""

from datetime import datetime
import logging
import os

import wandb

DEFAULT_WANDB_CONFIG = "~/.netrc"
DEFAULT_TAGS = ["tao-toolkit", "training"]
logger = logging.getLogger(__name__)

_WANDB_INITIALIZED = False


def is_wandb_initialized():
    """Check if wandb has been initialized."""
    global _WANDB_INITIALIZED  # pylint: disable=W0602,W0603
    return _WANDB_INITIALIZED


def check_wandb_logged_in():
    """Check if weights and biases have been logged in."""
    wandb_logged_in = False
    try:
        wandb_api_key = os.getenv("WANDB_API_KEY", None)
        if wandb_api_key is not None or os.path.exists(os.path.expanduser(DEFAULT_WANDB_CONFIG)):
            wandb_logged_in = wandb.login(key=wandb_api_key)
            return wandb_logged_in
    except wandb.errors.UsageError:
        logger.warning("WandB wasn't logged in.")
    return False


def initialize_wandb(project: str = "TAO Toolkit",
                     entity: str = None,
                     sync_tensorboard: bool = True,
                     save_code: bool = False,
                     notes: str = None,
                     tags: list = None,
                     name: str = "train",
                     config=None,
                     wandb_logged_in: bool = False,
                     results_dir: str = os.getcwd()):
    """Function to initialize wandb client with the weights and biases server.

    If wandb initialization fails, then the function just catches the exception
    and prints an error log with the reason as to why wandb.init() failed.

    Args:
        project (str): Name of the project to sync data with.
        entity (str): Name of the wanbd entity.
        sync_tensorboard (bool): Boolean flag to synchronize
            tensorboard and wanbd visualizations.
        notes (str): One line description about the wandb job.
        tags (list(str)): List of tags about the job.
        name (str): Name of the task running.
        config (OmegaConf.DictConf or Dict): Configuration element of the task that's being.
            Typically, this is the yaml container generated from the `experiment_spec`
            file used to run the job.
        wandb_logged_in (bool): Boolean flag to check if wandb was logged in.
        results_dir (str): Output directory of the experiment.

    Returns:
        No explicit returns.
    """
    logger.info("Initializing wandb.")
    try:
        assert wandb_logged_in, (
            "WandB client wasn't logged in. Please make sure to set "
            "the WANDB_API_KEY env variable or run `wandb login` in "
            "over the CLI and copy the ~/.netrc file to the container."
        )
        start_time = datetime.now()
        time_string = start_time.strftime("%d/%y/%m_%H:%M:%S")
        wandb_dir = os.path.join(results_dir, "wandb")
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        if tags is None:
            tags = DEFAULT_TAGS
        wandb_name = f"{name}_{time_string}"
        wandb.init(
            project=project,
            entity=entity,
            sync_tensorboard=sync_tensorboard,
            save_code=save_code,
            name=wandb_name,
            notes=notes,
            tags=tags,
            config=config
        )
        global _WANDB_INITIALIZED  # pylint: disable=W0602,W0603
        _WANDB_INITIALIZED = True
    except Exception as e:
        logger.warning("Wandb logging failed with error %s", e)
