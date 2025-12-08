import os
import pandas as pd
from core.generate import generate
from core.prompts import get_prompts
from shared.mlflow import (
    mlflow_start,
    mlflow_end,
    mlflow_active_model,
    mlflow_log_artifact,
    mlflow_log_dataset,
)
from shared.dataset import get_dataset
from shared.utils import timestamp
from shared.logger import get_logger

logger = get_logger(__name__)

FOLDER = "completions"


def merge_and_save(data: pd.DataFrame, completions: list[str], path: str = None):
    output_col = os.getenv("OUTPUT_COLUMN", "inference")
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


def inference(model_name: str, dataset_name: str, dataset_split: str = "eval", dataset_config: str = None):

    logger.info(f"🚀 Start inference of model {model_name} with dataset {dataset_name}")

    # Start mlflow run
    mlflow_start(model_name, run_type="inference", tags={"model_name": model_name, "dataset_name": dataset_name})
    mlflow_active_model()
    # mlflow_log_params()

    # Load dataset
    dataset, dataset_extras = get_dataset(
        dataset_name,
        dataset_split=dataset_split,
        dataset_config=dataset_config,
        # as_pandas=True,
    )
    mlflow_log_dataset(dataset_name, dataset, dataset_split=dataset_split)
    dataset = dataset.to_pandas()

    # Get prompts from dataset
    prompts = get_prompts(dataset)

    # Generate completions
    completions = generate(
        model_name,
        prompts=prompts,
        prompts_params=dataset_extras,
        sampling_params={},  # TODO: sampling params
    )

    # Write results
    output_file = merge_and_save(dataset, completions)

    # Log results as artifact
    mlflow_log_artifact(output_file)
    mlflow_end()

    logger.info(f"✅ Inference done! Results saved to {output_file}")
