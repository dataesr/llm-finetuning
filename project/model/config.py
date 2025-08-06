from transformers import AutoConfig
from huggingface_hub import hf_hub_download
from project.logger import get_logger

logger = get_logger(__name__)

FOLDER = "jobs"
MERGED_FOLDER = "merged"
CONFIG_DEFAULT = {"causallm": "openchat", "vision2seq": "nuextract"}
CONFIG_ALL = ["openchat", "nuextract"]


def model_get_config(model_name: str, forced_config: str = None) -> str:
    """
    Get model config

    Args:
    - model_name (str): model name
    - forced_config (str, optional): forced config. Defaults to None.

    Returns the corresponding config for the model
    """
    # Check if forced config exist
    if forced_config:
        if forced_config in CONFIG_ALL:
            logger.warning(f"Model {model_name} config manually forced to {forced_config}")
            return forced_config
        logger.warning(f"No forced config {forced_config} found")

    # Get config from model files
    config = AutoConfig.from_pretrained(model_name)
    if not config:
        logger.error(f"No remote config found for model {model_name}")
        raise ValueError(f"No remote config found for model {model_name}")

    # Check if model type exists in config
    model_type = config.model_type
    logger.debug(f"Model {model_name} type = {model_type}")
    if model_type in CONFIG_ALL:
        return model_type

    # Otherwise check model architecture
    model_arch = config.architectures[0]
    logger.debug(f"Model {model_name} architecture = {model_arch}")
    if model_arch and isinstance(model_arch, str):
        for conf in CONFIG_DEFAULT:
            if conf in model_arch.lower():
                return CONFIG_DEFAULT[conf]

    logger.error(f"No config found for model {model_name} (type={model_type}, arch={model_arch})")


def model_get_instruction(model_name: str) -> str:
    """
    Get instructions file from huggingface hub

    Args:
    - model_name (str): model repository

    Returns instructions as str
    """
    # Download file
    try:
        file_path = hf_hub_download(repo_id=model_name, filename="instruction.txt", repo_type="model")
    except Exception as error:
        logger.warning(f"⚠️ Instruction not found for model {model_name}: {error}")
        return None

    # Read file
    with open(file_path, "r", encoding="utf-8") as file:
        instruction = file.read()

    logger.debug(f"Found instruction for {model_name}: {instruction}")
    return instruction
