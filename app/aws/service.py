import subprocess
from app.logging import get_logger

logger = get_logger(__name__)


def sync(source: str, destination: str) -> bool:
    """Sync object from s3 bucket

    Args:
        source (str): Bucket or local object
        destination (str): Bucket or local object

    Returns:
        bool: True if sync succeeded, else False
    """
    is_ok = False
    command = ["aws", "s3", "sync", source, destination]
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Error: {result.stderr}")
    else:
        is_ok = True
        logger.debug(f"{result.stdout}")

    return is_ok
