import os
import json
from datasets import load_dataset, Dataset
from shared.utils import timestamp
from shared.logger import get_logger

logger = get_logger(name=__name__)

FOLDER = "datasets"
FOLDER_EXTRAS = "extras"

TEXT_FIELD = "text"
INSTRUCTION_FILENAME = "instruction.txt"
INSTRUCTION_FIELD = "instruction"
INPUT_FIELD = "input"
COMPLETION_FIELD = "completion"
CONVERSATIONS_FIELD = "messages"
CHAT_TEMPLATE_FIELD = "chat_template"
TEXT_FORMAT_FIELD = "text_format"

DEFAULT_TEXT_FORMAT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"


def get_file(object_name: str, check=False) -> str:
    """
    Get file path from object name

    Args:
    - object_name (str): ovh object name

    Returns:
    - file_path: object path
    """

    # Check folder exists
    if not os.path.isdir(FOLDER) and check:
        raise FileNotFoundError(f"Folder {FOLDER} not found on storage!")

    # Get file path
    file_path = f"{FOLDER}/{object_name}"

    # Check file exists
    if not os.path.isfile(file_path) and check:
        raise FileNotFoundError(f"File {file_path} not found on storage!")

    return file_path


def get_dataset(object_name: str, dataset_split: str = None, as_pandas: bool = False, **kwargs) -> Dataset:
    """
    Get a dataset from huggingface (or storage).

    Args:
    - object_name (str): huggingface path or ovh object_name
    - dataset_split (str): dataset split. Defaults to 'train'
    - as_pandas (bool): return DataFrame instead of Dataset

    Returns:
    - Dataset: dataset
    - Dict: dataset extras (prompts params)
    """
    # Get dataset split if needed
    split = dataset_split or "train"

    # Try to load from Hugging Face Hub
    try:
        logger.debug(f"Trying to load {object_name} from Hugging Face...")
        dataset = load_dataset(object_name, split=split)
    except:
        logger.debug(f"Trying to load from storage...")

        # Get file path
        file_path = get_file(object_name, check=True)

        # Load as dataset
        dataset = load_dataset("json", data_files={split: [file_path]}, split=split)

    if dataset:
        logger.debug(f"✅ Dataset {object_name} loaded!")
        logger.debug(f"Dataset schema: {dataset.features}")
        logger.debug(f"Dataset size: {len(dataset)}")
        logger.debug(f"Dataset sample: {dataset[0]}")
    else:
        logger.error(f"Error while loading {object_name}")
        raise Exception(f"Error while loading {object_name}")

    # Try to load dataset extras
    dataset_format = kwargs.get("dataset_format")
    dataset_config = kwargs.get("dataset_config")  # extras config name
    dataset_extras = get_dataset_extras(name=dataset_config, dataset_name=object_name)
    if dataset_format:
        if not dataset_extras:
            dataset_extras = {"dataset_format": dataset_format}
        elif dataset_extras.get("dataset_format"):
            logger.warning(f"Dataset format {dataset_format} already defined in dataset extras {dataset_extras}")
        else:
            dataset_extras["dataset_format"] = dataset_format

    # TODO: add randomness ?

    if as_pandas:
        return dataset.to_pandas(), dataset_extras

    return dataset, dataset_extras


def get_commit_hash(dataset: Dataset) -> str | None:
    """
    Retrieve commit hash from dataset checksums

    Args:
        dataset (Dataset): dataset

    Returns:
        commit_hash (str): dataset commit hash
    """
    commit_hash = None
    checksums = dataset.info.download_checksums
    if isinstance(checksums, dict) and checksums:
        checksums_list = list(checksums.keys())
        checksum_file = checksums_list[0].split("@")[1]
        commit_hash = checksum_file.split("/")[0]
    return commit_hash


def get_dataset_extras(name: str, dataset_name: str) -> dict:
    """
    Get prompts extra params from dataset

    Args:
        path_or_name (str): ovh file path
    Returns:
        extras (dict): extras json data
    """
    if not name:
        return None

    path = os.path.join(FOLDER_EXTRAS, dataset_name, name)
    if not path.endswith(".json"):
        path += ".json"
    path_on_disk = get_file(path, check=True)
    try:
        with open(path_on_disk, "r") as json_file:
            extras = json.load(json_file)
    except Exception as error:
        logger.error(f"Error parsing json from {path_on_disk}!")
        raise ValueError(str(error))

    logger.debug(f"Found dataset extras: {extras}")
    if extras:
        extras["dataset_config"] = extras.pop("name")
    return extras


def construct_one_conversation(user: str, system: str = None, assistant: str = None):
    """
    Construct a conversation from system, user and assistant messages

    Args:
    - user (str): user input
    - system (str, optional): system instructions. Defaults to None.
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
                    system=custom_instruction or example.get(INSTRUCTION_FIELD),
                    user=example[INPUT_FIELD],
                    assistant=example[COMPLETION_FIELD],
                )
            }
        else:
            # Non-conversational (Alpaca-style prompt-response text)
            instruction = custom_instruction or "You are an helpful assistant."
            text_format = custom_text_format if custom_text_format else DEFAULT_TEXT_FORMAT
            text = text_format.format(
                instruction=instruction, input=example[INPUT_FIELD], response=example[COMPLETION_FIELD]
            )
            return {prompts_field: text}

    dataset = dataset.map(map_conversations).select_columns([prompts_field])
    logger.debug(f"✅ Dataset formatted with {'conversation' if use_conversational_format else 'text'} format")
    logger.debug(f"Dataset columns: {dataset.column_names}")
    logger.debug(f"Dataset sample: {dataset[0][prompts_field]}")
    return dataset
