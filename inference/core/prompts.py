import os
import pandas as pd
from typing import Dict, Any
from shared.dataset import construct_one_conversation
from shared.logger import get_logger

logger = get_logger(__name__)


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
    instruction = prompts_params.get("instruction")
    text_format = prompts_params.get("text_format")
    chat_template = prompts_params.get("chat_template")

    if chat_template:
        tokenizer.chat_template = chat_template
        logger.warning(f"Using custom chat template : {tokenizer.chat_template}")

    # Format prompts
    formatted_prompts = prompts
    if tokenizer.chat_template:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                construct_one_conversation(system=instruction, user=prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    elif text_format:
        formatted_prompts = [
            text_format.format(instruction or "You are an helpful assistant.", prompt, "") for prompt in prompts
        ]
    else:
        pass  # No formatting

    logger.debug(f"Formatted prompts: {formatted_prompts[0]}")

    return formatted_prompts
