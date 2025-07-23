import os
import pandas as pd
from datasets import load_dataset, Dataset
from project.logger import get_logger

logger = get_logger(name=__name__)

FOLDER = "datasets"

TEXT_FIELD = "text"
INSTRUCTION_FIELD = "instruction"
INSTRUCTION_FILENAME = "instruction.txt"


def get_file(object_name: str) -> str:
    """
    Get file path from object name

    Args:
    - object_name (str): ovh object name

    Returns:
    - file_path: object path
    """

    # Check folder exists
    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(f"Folder {FOLDER} not found on storage!")

    # Get file path
    file_path = f"{FOLDER}/{object_name}"

    # Check file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found on storage!")

    return file_path


def get_dataset(object_name: str) -> Dataset:
    """
    Get a dataset from huggingface (or storage).

    Args:
    - object_name (str): huggingface path or ovh object_name

    Returns:
    - Dataset: dataset
    """

    # Try to load from Hugging Face Hub
    try:
        logger.debug(f"Trying to load {object_name} from Hugging Face...")
        dataset = load_dataset(object_name, split="train")
    except:
        logger.debug(f"Trying to load from storage...")

        # Get file path
        file_path = get_file(object_name)

        # Load as dataset
        dataset = load_dataset("json", data_files={"train": [file_path]}, split="train")

    if dataset:
        logger.debug(f"✅ Dataset {object_name} loaded!")
        logger.debug(f"Dataset schema: {dataset.features}")
        logger.debug(f"Dataset size: {len(dataset)}")
        logger.debug(f"Dataset sample: {dataset[0]}")
    else:
        logger.error(f"Error while loading {object_name}")
        raise Exception(f"Error while loading {object_name}")

    return dataset


def save_dataset_instruction(dataset, destination: str):
    """
    Save dataset instruction as file in model folder

    Args:
        dataset: Dataset used for fine tuning
        destination (str): Fine tuned model directory
    """
    # Get instruction from dataset
    instruction = dataset[0][INSTRUCTION_FIELD]

    # Save in model folder
    with open(f"{destination}/{INSTRUCTION_FILENAME}", "w") as file:
        file.write(instruction)

    logger.debug(f"✅ Instruction from dataset saved in {destination}")
