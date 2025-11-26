import os
from core.utils import model_get_finetuned_dir
from shared.hugging import upload_model_to_hub
from shared.logger import get_logger

logger = get_logger(__name__)


def model_push_to_hub(model_dir: str, check_repo: bool = False) -> str:
    """Push a model from storage to hugging face hub

    Args:
    - model_dir (str): model_directory
    - repo_id (str): The model repo ID on Hugging Face (e.g. "your-username/model-name")
    - private (bool): If True, creates a private repo

    Returns:
    - hf_hash (str): Hugging Face commit hash id
    """
    repo_id = os.getenv("HF_PUSH_REPO")

    if not repo_id:
        if check_repo:
            raise ValueError(f"Env var 'HF_PUSH_REPO' not defined!")
        return None

    logger.info(f"Pushing model ({model_dir}) to HF ({repo_id})...")

    # Get model folder
    model_dir = model_get_finetuned_dir(model_dir, check=True)

    # Upload model to hub
    hf_hash = upload_model_to_hub(model_dir, repo_id=repo_id, private=True)
    if hf_hash:
        logger.info(f"Successfully pushed model to huggingface")

    return hf_hash
