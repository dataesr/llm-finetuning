from transformers import AutoConfig
from project.logger import get_logger

logger = get_logger(__name__)

FOLDER = "jobs"
MERGED_FOLDER = "merged"
CONFIG_DEFAULT = {"causallm": "llama", "vision2seq": "qwen2_vl"}
CONFIG_ALL = ["llama", "qwen2_vl"]


def model_get_config(model_name: str, forced_config: str = None) -> str:
    # Check if forced config exist
    if forced_config:
        if forced_config in CONFIG_DEFAULT:
            logger.warning(f"Model {model_name} config manually forced to {CONFIG_DEFAULT[forced_config]}")
            return CONFIG_DEFAULT[forced_config]
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
