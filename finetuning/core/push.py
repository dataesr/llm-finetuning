from core.utils import model_get_finetuned_dir
from shared.hugging import upload_model_to_hub


def model_push_to_hub(model_dir: str, repo_id: str, private=False) -> str:
    """Push a model from storage to hugging face hub

    Args:
    - model_dir (str): model_directory
    - repo_id (str): The model repo ID on Hugging Face (e.g. "your-username/model-name")
    - private (bool): If True, creates a private repo

    Returns:
    - hf_hash (str): Hugging Face commit hash id
    """
    # Get model folder
    model_dir = model_get_finetuned_dir(model_dir, check=True)

    # Upload model to hub
    hf_hash = upload_model_to_hub(model_dir, repo_id=repo_id, private=private)
    return hf_hash
