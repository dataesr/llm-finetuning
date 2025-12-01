import os
import pandas as pd
from typing import Dict, Any
from shared.dataset import (
    construct_one_conversation,
    TEXT_FORMAT_FIELD,
    INSTRUCTION_FIELD,
    CHAT_TEMPLATE_FIELD,
    DEFAULT_TEXT_FORMAT,
)
from shared.utils import should_use_conversational_format
from shared.logger import get_logger

logger = get_logger(__name__)


# TODO
def validate_prompt(prompt, tokenizer, max_input_tokens=7500):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_input_tokens:
        raise ValueError(f"Prompt is {len(tokens)} tokens, max is {max_input_tokens}")


def get_prompts(data: pd.DataFrame) -> list[str]:
    input_col = os.getenv("INPUT_COLUMN", "input")
    if input_col not in data.columns:
        raise ValueError(f"Column {input_col} not found on data! Set env var 'INPUT_COLUMN' to select the column name.")

    prompts = data[input_col].astype(str).to_list()
    return prompts


def apply_chat_template(tokenizer, prompts: list[str], prompts_params: Dict[str, Any]) -> list[str]:
    logger.debug(f"Formatting {len(prompts)} prompts")
    if prompts_params:
        logger.debug(f"Custom formatting params: {prompts_params}")

    # Get prompts param
    instruction = prompts_params.get(INSTRUCTION_FIELD, "")
    dataset_format = prompts_params.get("dataset_format")
    text_format = prompts_params.get(TEXT_FORMAT_FIELD, DEFAULT_TEXT_FORMAT)
    chat_template = prompts_params.get(CHAT_TEMPLATE_FIELD)

    if chat_template:
        tokenizer.chat_template = chat_template
        logger.warning(f"Using custom chat template : {tokenizer.chat_template}")

    # Format prompts
    formatted_prompts = prompts
    use_conversational_format = should_use_conversational_format(dataset_format, tokenizer.chat_template)
    if use_conversational_format:
        # Conversational format (list of messages, ChatML-like)
        formatted_prompts = [
            tokenizer.apply_chat_template(
                construct_one_conversation(system=instruction, user=prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    else:
        # Non-conversational (Alpaca-style prompt-response text)
        formatted_prompts = [text_format.format(instruction=instruction, input=prompt, response="") for prompt in prompts]

    logger.debug(f"Formatted prompts: {formatted_prompts[0]}")

    return formatted_prompts
