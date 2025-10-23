import os
import shutil
from shared.logger import get_logger

logger = get_logger(__name__)


def reset_folder(dir_path: str, delete=False):
    """
    Delete contents of a folder

    Args:
    - dir_path (str): folder path
    - delete (bool): Delete the folder if True, else only delete contents
    """

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    else:
        logger.error(f"Folder {dir_path} not found on storage!")

    if not delete:
        os.makedirs(dir_path)


def should_use_conversational_format(dataset_format_arg: str = None, dataset_chat_template=None):
    if dataset_format_arg == "conversational":
        logger.debug("Format set to 'conversational'")
        return True
    elif dataset_format_arg == "text":
        logger.debug("Format set to 'text'")
        return False
    else:
        if dataset_chat_template is not None:
            logger.debug("Format automatically set to 'conversational'")
            return True
        else:
            logger.debug("Format automatically set to 'text'")
            return False
