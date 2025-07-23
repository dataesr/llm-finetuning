import os
from project.model.config import FOLDER
from project.logger import get_logger
from project._utils import reset_folder

logger = get_logger(__name__)

def model_default_output_name(model_name: str, suffix: str = None) -> str:
    """
    Get default output name from base model name

    Args:
    - model_name (str): Base model name
    - suffix (str): Suffix to add

    Returns:
    - output_model_name (str): New model name
    """
    output_model_name = f"{model_name.split("/")[-1]}-{suffix}"
    return output_model_name

def model_get_finetuned_dir(output_model_name:str):
    """
    Get finetuned model folder

    Args:
    - output_model_name (str): Fine-tuned model name

    Returns:
    - model_dir (str): Fine-tuned model path
    """
    from project.model.config import FOLDER, MERGED_FOLDER

    model_dir = os.path.join(FOLDER, output_model_name, MERGED_FOLDER)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Folder {model_dir} not found on storage!")

    return model_dir

def model_delete_dir(output_model_name: str):
    """
    Delete all model files

    Args:
    - model_dir (str): model folder
    """
    model_dir = os.path.join(FOLDER, output_model_name)

    try:
        reset_folder(model_dir, delete=True)
        logger.info(f"✅ Model folder {model_dir} deleted")
    except Exception as error:
        logger.debug(f"Cannot delete folder {model_dir}: {error}")