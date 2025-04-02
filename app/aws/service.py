import os
from app._utils import cmd
from app.logging import get_logger

logger = get_logger(__name__)


def sync(source: str, destination: str, is_directory=False) -> bool:
    """Sync an object from/to an s3 bucket

    Args:
        source (str): Bucket or local object
        destination (str): Bucket or local object
        is_directory (bool, optional): Specify if the object to sync is a directory. Defaults to False.

    Returns:
        bool: True if sync succeeded, else False
    """
    s3_cmd = "sync" if is_directory else "cp"
    command = ["aws", "s3", s3_cmd, source, destination]
    is_ok = cmd.run(command)

    return is_ok


def upload(file_or_dir_name: str, bucket_name: str, object_name=None, is_directory=False) -> bool:
    """Upload a file or a folder to an s3 bucket

    Args:
        file_or_dir_name (str): File or folder to upload
        bucket (str): Bucket to upload to
        object_name (str, optional): S3 object name. If not specified then file_or_dir_name is used

    Returns:
        bool: True if file was uploaded, else False
    """
    logger.debug(f"Uploading {file_or_dir_name} in {bucket_name} as {object_name}")

    # If object_name was isnt specified, use file_or_dir_name
    if object_name is None:
        object_name = os.path.basename(file_or_dir_name)

    # Define source and destination
    source = file_or_dir_name
    destination = f"s3://{bucket_name}/{object_name}"

    # Sync with object storage
    is_uploaded = sync(source, destination, is_directory)

    return is_uploaded


def download(object_name: str, bucket_name: str, file_or_dir_name: str, is_directory=False) -> bool:
    """Download a file or a folder from an s3 bucket

    Args:
        object_name (str): S3 object name (could be a file or a folder)
        bucket (str): Bucket to download from
        file_or_dir_name (str): File or folder to create

    Returns:
        bool: True if file was downloaded, else False
    """
    logger.debug(f"Downloading {object_name} from {bucket_name} as {file_or_dir_name}")

    # Define source and destination
    source = f"s3://{bucket_name}/{object_name}"
    destination = file_or_dir_name

    # Sync with object storage
    is_uploaded = sync(source, destination, is_directory)

    return is_uploaded
