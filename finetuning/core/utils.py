import os
from core.config import FOLDER, CHECKPOINTS_FOLDER, MERGED_FOLDER, EXTRACTED_FOLDER
from shared.logger import get_logger
from shared.utils import create_folder, reset_folder

logger = get_logger(__name__)

def hf_get_push_repo(check:bool=False):
    hf_repo = os.getenv("HF_PUSH_REPO")
    if check and not hf_repo:
        raise ValueError(f"Env var 'HF_PUSH_REPO' not defined!") 

def model_default_output_name(model_name: str, suffix: str = None) -> str:
    """
    Get default output name from base model name

    Args:
    - model_name (str): Base model name
    - suffix (str): Suffix to add

    Returns:
    - model_dir (str): model directory
    """
    model_output_name = f"{model_name.split("/")[-1]}"
    if suffix:
        model_output_name += suffix
    return model_output_name


def model_initialize_dir(model_name: str) -> tuple:
    """
    Initialize model output folder and name

    Args:
    - model_name (str): Base model to finetune

    Returns:
    - model_dir (str): Model training directory
    """
    # Default output model name
    model_output_name = model_default_output_name(model_name)

    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(f"Folder {FOLDER} not found on storage!")

    # Create output folder
    dir_path = create_folder(f"{FOLDER}/{model_name}")
    model_dir = dir_path.removeprefix(f"{FOLDER}/")

    return model_output_name, model_dir

def model_get_output_dir(model_dir:str, check:bool=False) -> str:
    """
    Get output model folder

    Args:
    - model_dir (str): Model directory

    Returns:
    - finetuned_dir (str): Model training output directory
    """
    output_dir = os.path.join(FOLDER, model_dir)

    if check and not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Folder {output_dir} not found on storage!")

    return output_dir

def model_get_checkpoints_dir(model_dir:str, check:bool=False) -> str:
    """
    Get checkpoints model folder

    Args:
    - model_dir (str): Model directory

    Returns:
    - checkpoints_dir (str): Model training checkpoints directory
    """
    checkpoints_dir = os.path.join(FOLDER, model_dir, CHECKPOINTS_FOLDER)

    if check and not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Folder {checkpoints_dir} not found on storage!")

    return checkpoints_dir

def model_get_finetuned_dir(model_dir:str, check:bool=False) -> str:
    """
    Get finetuned model folder

    Args:
    - model_dir (str): Model directory

    Returns:
    - finetuned_dir (str): Model finetuned directory
    """
    finetuned_dir = os.path.join(FOLDER, model_dir, MERGED_FOLDER)

    if check and not os.path.isdir(finetuned_dir):
        raise FileNotFoundError(f"Folder {finetuned_dir} not found on storage!")

    return finetuned_dir

def model_get_extracted_dir(model_dir:str, check:bool=False) -> str:
    """
    Get finetuned model folder

    Args:
    - model_dir (str): Model directory

    Returns:
    - model_dir (str): Fine-tuned model path
    """
    extracted_dir = os.path.join(FOLDER, model_dir, EXTRACTED_FOLDER)

    if check and not os.path.isdir(extracted_dir):
        raise FileNotFoundError(f"Folder {extracted_dir} not found on storage!")

    return extracted_dir

def model_delete_dir(model_dir: str):
    """
    Delete all model files

    Args:
    - model_dir (str): model folder
    """
    dir_path = model_get_output_dir(model_dir)

    try:
        reset_folder(dir_path, delete=True)
        logger.info(f"✅ Model folder {dir_path} deleted")
    except Exception as error:
        logger.debug(f"Cannot delete folder {dir_path}: {error}")
        
        
        
def get_env(var_name, default_value, caster):
    """
    Read an environment variable and cast it, falling back to default on missing/invalid values.

    Args:
    - var_name: environment variable name
    - default_value: fallback value when env var is missing
    - caster: callable to cast the string value (e.g., int, float)
    """
    value = os.getenv(var_name)
    if value is None:
        return default_value
    try:
        return caster(value)
    except Exception:
        logger.warning(f"⚠️ Invalid value for {var_name}='{value}', using default {default_value}")
        return default_value