import os
from app.logging import get_logger
from app.aws import service as aws_service

logger = get_logger(__name__)


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an s3 bucket

    Args:
        file_name (str): File to upload
        bucket (str): Bucket to upload to
        object_name (str, optional): S3 object name. If not specified then file_name is used

    Returns:
        bool: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    logger.debug(f"Uploading {file_name} in {bucket} as {object_name}")

    source = file_name
    destination = f"s3://{bucket}/{object_name}"
    is_uploaded = aws_service.sync(source, destination)

    return is_uploaded


def download_file(bucket, object_name, file_name):
    """Download a file from an s3 bucket

    Args:
        bucket (str): Bucket to upload to
        object_name (str): S3 object name
        file_name (str): File to create

    Returns:
        bool: True if file was downloaded, else False
    """
    logger.debug(f"Downloading {object_name} from {bucket} to {file_name}")

    source = f"s3://{bucket}/{object_name}"
    destination = file_name
    is_downloaded = aws_service.sync(source, destination)

    return is_downloaded
