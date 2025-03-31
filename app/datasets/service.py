import os
import pandas as pd
from app.object_storage import service as os_service
from app.logging import get_logger

logger = get_logger(name=__name__)

BUCKET = "llm-datasets"
FOLDER = "datasets"


def create_folder():
    """Create datasets folder"""
    is_folder = os.path.isdir(FOLDER)
    if not is_folder:
        os.makedirs(FOLDER)


def load(object_name: str) -> pd.DataFrame:
    """Get a dataset from storage. Download from object_storage if not found.

    Args:
        object_name (str): S3 object_name

    Returns:
        DataFrame: dataset
    """
    # Check folder exists
    create_folder()

    file_name = f"{FOLDER}/{object_name}"

    # Check file exists
    is_file = os.path.isfile(file_name)
    logger.debug(f"is_file: {is_file}")

    # Download file if not exits
    if not is_file:
        logger.debug(f"File {object_name} not found, try to download it from object storage")
        is_downloaded = os_service.download_file(bucket=BUCKET, object_name=object_name, file_name=file_name)
        logger.debug(f"is_downloaded: {is_downloaded}")

        if not is_downloaded:
            logger.error(f"File {object_name} not downloaded, exit..")
            return None

    # Load as DataFrame
    dataset = pd.read_json(file_name)

    return dataset
