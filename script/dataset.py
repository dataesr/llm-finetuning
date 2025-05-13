import os
import pandas as pd
from datasets import load_dataset, Dataset
from logger import get_logger

logger = get_logger(name=__name__)

FOLDER = "datasets"

TEXT_FIELD = "text"
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


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


def get_dataset(object_name: str, eos_token) -> Dataset:
    """
    Get a dataset from storage.

    Args:
    - object_name (str): ovh object_name
    - eos_token: tokenizer end of sentence token

    Returns:
    - Dataset: dataset
    """
    # Get file path
    file_path = get_file(object_name)

    # Formatting function
    def formatting_prompts_func(samples):
        instructions = samples["instruction"]
        inputs = samples["prompt"]
        outputs = samples["completion"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + eos_token
            texts.append(text)
        return {
            TEXT_FIELD: texts,
        }

    pass

    # Load as dataset
    dataset = load_dataset("json", data_files={"train": [file_path]}, split="train")

    # Format dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)

    if dataset:
        logger.debug(f"✅ Dataset {object_name} loaded!")
        logger.debug(f"Dataset schema: {dataset.features}")
        logger.debug(f"Dataset sample: {dataset[0]}")
    else:
        logger.error(f"Error while loading {file_path}")

    return dataset
