import subprocess
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
    is_ok = False
    s3_cmd = "sync" if is_directory else "cp"
    command = ["aws", "s3", s3_cmd, source, destination]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Error: {result.stderr}")
    else:
        is_ok = True
        logger.debug(f"{result.stdout}")

    return is_ok
