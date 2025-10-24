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


def create_folder(dir_path: str, override=False) -> str:
    """
    Create a directory with smart handling of existing paths.

    Args:
        dir_path (str): The directory path to create
        override (bool): If True, override existing non-empty directory

    Returns:
        str: The actual path that was created

    Behavior:
        - Creates directory if path doesn't exist (including nested paths)
        - Does nothing if path exists and is empty
        - Creates directory with different name if path exists and is non-empty
        - Overrides existing directory if override=True
    """
    # Create dir if path doesnt exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return dir_path

    # Folder exists
    if os.path.isdir(dir_path):

        # Folder is empty
        if not os.listdir(dir_path):
            return dir_path

        # Folder not empty - override
        if override:
            reset_folder(dir_path)
            return dir_path

        # Folder not empty - create with different name
        counter = 1
        while True:
            new_dir_path = f"{dir_path}_{counter}"
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
                return new_dir_path
            counter += 1

    # force creation by default
    reset_folder(dir_path)
    return dir_path


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
