import os
import json
from huggingface_hub import create_repo, upload_folder, hf_hub_download
from project.model.utils import model_get_finetuned_dir
from project.logger import get_logger

logger = get_logger(__name__)


def download_file_from_hub(filename: str, repo_id: str, repo_type: str, local_dir: str):
    """
    Downloads a specific file from a Hugging Face Hub repository.

    Args:
    - repo_id (str): The model repo ID on Hugging Face (e.g. "your-username/model-name")
    - filename (str): The name of the file to download (e.g. "config.json", "pytorch_model.bin")
    - repo_type (str): Huggingface repo type: model or dataset
    - local_dir (str): The local directory to save the downloaded file
    """
    logger.info(f"Start downloading {filename} from https://huggingface.co/{repo_id}")

    # Download the file
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=local_dir,
    )
    logger.info(f"✅ File downloaded to {local_path}")
    return local_path


def get_json_from_hub(filename: str, repo_id: str, repo_type: str):
    json_data = None

    if not filename.endswith(".json"):
        logger.debug(f"{filename} not a json file; cancel download.")
        return json_data

    try:
        json_path = download_file_from_hub(filename=filename, repo_id=repo_id, repo_type=repo_type, local_dir=None)
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
    except Exception as error:
        logger.error(f"Error downloading {filename} from {repo_id}")
        return json_data

    logger.debug(f"Successfully loaded {filename} from {repo_id}")
    return json_data


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
    model_dir = model_get_finetuned_dir(output_model_name, check=True)

    # Upload model to hub
    upload_model_to_hub(model_dir, repo_id=repo_id, private=private)
