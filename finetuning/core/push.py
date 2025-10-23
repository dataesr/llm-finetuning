from core.utils import model_get_finetuned_dir
from shared.hugging import upload_model_to_hub


def model_push_to_hub(output_model_name: str, repo_id: str, private=False):
    """Push a model from storage to hugging face hub

    Args:
    - output_model_name (str): Name of the model to push
    - repo_id (str): The model repo ID on Hugging Face (e.g. "your-username/model-name")
    - private (bool): If True, creates a private repo
    """
    # Get model folder
    model_dir = model_get_finetuned_dir(output_model_name, check=True)

    # Upload model to hub
    upload_model_to_hub(model_dir, repo_id=repo_id, private=private)
