import os
import json
from datasets import load_dataset, Dataset
from project.hugging import get_json_from_hub
from project.logger import get_logger

logger = get_logger(name=__name__)

FOLDER = "datasets"

TEXT_FIELD = "text"
INSTRUCTION_FILENAME = "instruction.txt"
INSTRUCTION_FIELD = "instruction"
INPUT_FIELD = "input"
COMPLETION_FIELD = "completion"
CONVERSATIONS_FIELD = "messages"
CHAT_TEMPLATE_FIELD = "chat_template"
TEXT_FORMAT_FIELD = "text_format"

DEFAULT_TEXT_FORMAT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"

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

    # TODO: add randomness

    return dataset


def get_dataset_extras(repo_id: str) -> dict:
    """
    Get extras from huggingface dataset

    Args:
        repo_id (str): Huggingface dataset repository
    Returns:
        extras (dict): extras json data
    """
    extras = get_json_from_hub(filename="extras.json", repo_id=repo_id, repo_type="dataset")
    logger.debug(f"extras: {extras}")
    return extras


def save_dataset_extras(extras: dict, destination: str):
    """
    Save dataset extras as json in model folder

    Args:
        extras (dict): Extras from dataset
        destination (str): Fine tuned model directory
    """
    if not extras:
        logger.debug("No extras to save")
        return

    # Save in model folder
    with open(f"{destination}/extras.json", "w") as file:
        json.dump(extras, file, indent=4)

    logger.debug(f"✅ Extras from dataset saved in {destination}")


def save_dataset_instruction(dataset: Dataset, destination: str):
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


def construct_one_conversation(system: str, user: str, assistant: str = None):
    """
    Construct a conversation from system, user and assistant messages

    Args:
    - system (str): system instructions
    - user (str): user input
    - assistant (str, optional): assistant completion for training. Defaults to None.

    Returns a conversation object
    """
    conversation = []

    # Add system prompt
    if system:
        conversation.append({"role": "system", "content": system})

    # Add user prompt
    conversation.append({"role": "user", "content": user})

    # Add assistant prompt
    if assistant:
        conversation.append({"role": "assistant", "content": assistant})

    return conversation


def construct_prompts(
    dataset: Dataset,
    custom_instruction: str = None,
    custom_text_format: str = None,
    use_conversational_format: bool = False,
) -> Dataset:
    """
    Construct prompts for training on a dataset

    Args:
    - dataset (Dataset): training dataset
    - custom_instruction (str): custom system prompt
    - custom_text_format (str): custom text format
    - use_conversational_format (bool): if True, use conversational format

    Returns the training dataset with a conversations column
    """
    prompts_field = CONVERSATIONS_FIELD if use_conversational_format else TEXT_FIELD

    def map_conversations(example):
        if use_conversational_format:
            # Conversational format (list of messages, ChatML-like)
            return {
                prompts_field: construct_one_conversation(
                    system=custom_instruction,
                    user=example[INPUT_FIELD],
                    assistant=example[COMPLETION_FIELD],
                )
            }
        else:
            # Non-conversational (Alpaca-style prompt-response text)
            instruction = custom_instruction or "You are an helpful assistant."
            text_format = custom_text_format if custom_text_format else DEFAULT_TEXT_FORMAT
            text = text_format.format(instruction, example[INPUT_FIELD], example[COMPLETION_FIELD])
            return {prompts_field: text}

    dataset = dataset.map(map_conversations).select_columns([prompts_field])
    logger.debug(f"✅ Dataset formatted with {'conversation' if use_conversational_format else 'text'} format")
    logger.debug(f"Dataset columns: {dataset.column_names}")
    logger.debug(f"Dataset sample: {dataset[0][prompts_field]}")
    return dataset
