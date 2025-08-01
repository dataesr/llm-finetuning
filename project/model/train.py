import os
import importlib
from project.model.config import model_get_config, FOLDER
from project.model.utils import model_default_output_name
from project.dataset import get_dataset
from project._utils import reset_folder
from project.logger import get_logger

logger = get_logger(__name__)


def initialize(model_name: str, output_model_name=None) -> tuple:
    """
    Initialize llm folder

    Args:
    - model_name (str): Base model to finetune
    - output_model_name (str): Finetuned model name. Default to None

    Returns:
    - output_model_name (str): Finetuned model name
    - output_dir (str): Finetuned model directory
    """
    # Default model name
    if not output_model_name:
        output_model_name = model_default_output_name(model_name, "train")

    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(f"Folder {FOLDER} not found on storage!")

    # Reset output folder
    output_dir = f"{FOLDER}/{output_model_name}"
    reset_folder(output_dir)

    return output_model_name, output_dir


def model_train(model_name: str, dataset_name: str, output_model_name: str = None) -> str:
    logger.info(f"🚀 Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Initialize llm folder
    output_model_name, output_dir = initialize(model_name, output_model_name)

    # Load dataset
    dataset = get_dataset(dataset_name)

    # Get pipeline
    config = model_get_config(model_name)
    pipeline = importlib.import_module(f"project.pipeline.{config}")

    # Train model
    pipeline.train(
        model_name=model_name,
        output_model_name=output_model_name,
        output_dir=output_dir,
        dataset=dataset
    )

    logger.info(f"Fine tuning of model {model_name} done")
    return output_model_name
