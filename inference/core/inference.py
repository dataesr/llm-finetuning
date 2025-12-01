import os
import argparse
import pandas as pd
from mlflow import MlflowClient
from typing import List, Dict, Any, Optional
from core.generate import load_engine, generate
from core.prompts import get_prompts
from core.completions import merge_and_save
from shared.mlflow import mlflow_start, mlflow_end, mlflow_set_tags, mlflow_active_model, mlflow_log_artifact
from shared.dataset import get_dataset
from shared.logger import get_logger

logger = get_logger(__name__)


def inference(model_name: str, dataset_name: str, dataset_split: str = "eval", dataset_config: str = None):

    logger.info(f"🚀 Start inference of model {model_name} with dataset {dataset_name}")

    # Start mlflow run
    mlflow_start(model_name)
    mlflow_set_tags({"model_name": model_name, "dataset_name": model_name})
    mlflow_active_model()
    # mlflow_log_params()
    # mlflow_active_model()

    # Load dataset
    dataset, dataset_extras = get_dataset(
        dataset_name,
        dataset_split=dataset_split,
        dataset_config=dataset_config,
        as_pandas=True,
    )
    mlflow_set_tags({"dataset_name": dataset_name, "dataset_split": dataset_split})
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
