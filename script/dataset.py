import os
import pandas as pd
from datasets import load_dataset, Dataset
from script.logger import get_logger

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


def get_dataset(object_name: str, tokenizer, use_chatml: bool = False) -> Dataset:
    """
    Get a dataset from storage.

    Args:
    - object_name (str): ovh object_name
    - tokenizer: tokenizer
    - use_chatml (bool): if True, use chatml tokenizer

    Returns:
    - Dataset: dataset
    """

    if use_chatml:
        # Formatting function
        def formatting_prompts_func(samples):
            conversations = samples["chat_ml_format"]
            texts = [
                tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations
            ]
            return {
                "text": texts,
            }

        pass

    else:
        # Formatting function
        def formatting_prompts_func(samples):
            instructions = samples["instruction"]
            inputs = samples["input"]
            outputs = samples["completion"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
                texts.append(text)
                return {
                    TEXT_FIELD: texts,
                }

        pass

    # Try to load from Hugging Face Hub
    try:
        logger.debug(f"Trying to load {object_name} from Hugging Face...")
        dataset = load_dataset(object_name, split="train")
    except Exception as error:
        logger.debug(f"Trying to load from storage...")

        # Get file path
        file_path = get_file(object_name)

        # Load as dataset
        dataset = load_dataset("json", data_files={"train": [file_path]}, split="train")

    # Format dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)

    if dataset:
        logger.debug(f"✅ Dataset {object_name} loaded!")
        logger.debug(f"Dataset schema: {dataset.features}")
        logger.debug(f"Dataset sample: {dataset[0][TEXT_FIELD]}")
    else:
        logger.error(f"Error while loading {file_path}")

    return dataset
