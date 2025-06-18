import os
import shutil
from script.logger import get_logger

logger = get_logger(__name__)

def get_default_output_name(model_name: str) -> str:
    """
    Get default output name from base model name

    Args:
    - model_name (str): Base model name

    Returns:
    - output_model_name (str): New model name
    """
    output_model_name = f"{model_name.split("/")[-1]}-finetuned"
    return output_model_name

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