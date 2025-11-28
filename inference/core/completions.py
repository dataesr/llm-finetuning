import os
import pandas as pd
from shared.utils import timestamp
from shared.logger import get_logger

logger = get_logger(__name__)

FOLDER = "completions"


def merge_and_save(data: pd.DataFrame, completions: list[str], path: str = None):
    output_col = os.getenv("OUTPUT_COLUMN", "completion")
    if output_col in data.columns:
        logger.warning(f"Existing column '{output_col}' will be overridden by generated completions!")

    # Check completions
    if not isinstance(completions, list):
        raise TypeError(f"Generated completions must be a list, got {type(completions)}")

    if len(completions) != len(data):
        logger.error(f"Generated {len(completions)} completions from {len(data)} texts, only completions will be saved")
        output = pd.DataFrame.from_dict({output_col: completions})
    else:
        logger.info(f"✅ Generated {len(completions)}")
        output = data.copy()
        output[output_col] = pd.Series(completions)

    # Save to JSON
    output_path = path or f"completions_{timestamp()}.json"
    output_path = os.path.join(FOLDER, output_path)
    if not output_path.endswith(".json"):
        output_path += ".json"
    output.to_json(output_path, orient="records")

    return output_path
