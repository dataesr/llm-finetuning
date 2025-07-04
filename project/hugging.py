import os
from project.logger import get_logger
from huggingface_hub import create_repo, upload_folder

logger = get_logger(__name__)


def get_model(output_model_name: str) -> str:
    """
    Get model folder

    Args:
    - output_model_name (str): Fine-tuned model name

    Returns:
    - model_dir (str): Fine-tuned model path
    """
    from project.pipeline import FOLDER, MERGED_FOLDER

    model_dir = os.path.join(FOLDER, output_model_name, MERGED_FOLDER)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Folder {model_dir} not found on storage!")

    return model_dir


def upload_model_to_hub(model_dir: str, repo_id: str, private=False):
    """
    Uploads a model directory to the Hugging Face Hub.

    Args:
    - model_dir (str): Path to the saved model folder (should include config.json, pytorch_model.bin, tokenizer, etc.)
    - repo_id (str): The model repo ID on Hugging Face
    - private (bool): If True, creates a private repo
    """

    logger.info(f"Start uploading model from {model_dir} to https://huggingface.co/{repo_id}")

    # Get hugging face token from env
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("'HF_TOKEN' not found in env. Skip upload to hub.")
        return

    # Create the repo if it doesn't exist
    repo_url = create_repo(repo_id, private=private, token=token, exist_ok=True)
    logger.debug(f"repo_url = {repo_url}")

    # Upload all contents of the folder
    commit_info = upload_folder(
        folder_path=model_dir,
        path_in_repo=".",  # Upload directly into the repo root
        repo_id=repo_id,
        token=token,
    )
    logger.debug(f"commit_info = {commit_info}")

    logger.info(f"✅ Model uploaded to https://huggingface.co/{repo_id}")


def push_to_hub(output_model_name: str, repo_id: str, private=False):
    """Push a model from storage to hugging face hub

    Args:
    - output_model_name (str): Name of the model to push
    - repo_id (str): The model repo ID on Hugging Face (e.g. "your-username/model-name")
    - private (bool): If True, creates a private repo
    """
    # Get model folder
    model_dir = get_model(output_model_name)

    # Upload model to hub
    upload_model_to_hub(model_dir, repo_id=repo_id, private=private)
