import os
import pandas as pd
from datasets import load_dataset, Dataset
from logger import get_logger

logger = get_logger(name=__name__)

FOLDER = "datasets"


def get_file(object_name: str) -> str:
    """
    Get file path from object name

    Args:
    - object_name (str): ovh object name

    Returns:
    - file_path: object path
    """

    # Check folder exists
    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(f"Folder {FOLDER} not found on storage!")

    # Get file path
    file_path = f"{FOLDER}/{object_name}"

    # Check file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found on storage!")

    return file_path


def get_dataset(object_name: str) -> Dataset:
    """
    Get a dataset from storage.

    Args:
    - object_name (str): ovh object_name

    Returns:
    - Dataset: dataset
    """
    # Get file path
    file_path = get_file(object_name)

    # Load as dataset
    dataset = load_dataset("json", data_files={"train": [file_path]}, split="train")
    if dataset:
        logger.debug(f"✅ Dataset {object_name} loaded!")
    else:
        logger.error(f"Error while loading {file_path}")

    return dataset
