import os
import pandas as pd
from datasets import load_dataset, Dataset
from logger import get_logger

logger = get_logger(name=__name__)

BUCKET = "llm-datasets"
FOLDER = "datasets"


def initialize(object_name: str):
    """Initialize datasets folder"""
    is_folder = os.path.isdir(FOLDER)

    # Check folder exists
    if not is_folder:
        logger.error(f"Folder {FOLDER} not found on storage!")

    # Get file name
    file_name = f"{FOLDER}/{object_name}"

    return file_name


def get_dataset(object_name: str) -> Dataset:
    """Get a dataset from storage.

    Args:
        object_name (str): S3 object_name

    Returns:
        DataFrame: dataset
    """
    # Initialize job
    file_name = initialize(object_name)

    # Check file exists
    is_file = os.path.isfile(file_name)

    if not is_file:
        logger.error(f"File {file_name} not found on storage!")
        return None

    # Load as dataset
    dataset = load_dataset("json", data_files={"train": [file_name]}, split="train")
    if dataset:
        logger.debug(f"Dataset {file_name} loaded!")

    return dataset
