import os
import pandas as pd
from datasets import load_dataset, Dataset
from app.aws import service as aws_service
from app.logging import get_logger

logger = get_logger(name=__name__)

BUCKET = "llm-datasets"
FOLDER = "datasets"


def initialize(object_name: str):
    """Initialize datasets folder"""
    is_folder = os.path.isdir(FOLDER)
    if not is_folder:
        os.makedirs(FOLDER)

    file_name = f"{FOLDER}/{object_name}"
    return file_name


def load(object_name: str) -> Dataset:
    """Get a dataset from storage. Download from object_storage if not found.

    Args:
        object_name (str): S3 object_name

    Returns:
        DataFrame: dataset
    """
    # Initialize job
    file_name = initialize(object_name)

    # Check file exists
    is_file = os.path.isfile(file_name)

    # Download file if not exits
    if not is_file:
        logger.debug(f"File {object_name} not found, try to download it from object storage")
        is_downloaded = aws_service.download(object_name, BUCKET, file_name)

        if not is_downloaded:
            logger.error(f"File {object_name} not downloaded, exit..")
            return None

    # Load as dataset
    dataset = load_dataset("json", data_files={"train": [file_name]}, split="train")
    logger.debug(f"Dataset {file_name} loaded!")

    return dataset
