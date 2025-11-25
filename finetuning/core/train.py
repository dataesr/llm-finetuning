import importlib
import torch
import mlflow
from core.mlflow import mlflow_start, mlflow_end, mlflow_log_dataset, mlflow_log_prompts_params
from shared.dataset import get_dataset, get_dataset_extras
from shared.logger import get_logger

logger = get_logger(__name__)


def model_train(model_name: str, model_dir: str, pipeline_name: str, dataset_name: str, **kwargs) -> str:
    logger.info(f"🚀 Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Cleanup
    torch.cuda.empty_cache()

    # Start mlflow
    mlflow_start()

    # Load dataset
    dataset = get_dataset(dataset_name)
    dataset_format = kwargs.get("dataset_format")
    dataset_extras = get_dataset_extras(kwargs.get("dataset_config"), dataset_name)
    mlflow_log_dataset(dataset_name, dataset)
    mlflow_log_prompts_params(dataset_extras)

    # Get pipeline
    pipeline = importlib.import_module(f"core.pipeline.{pipeline_name}")

    # Train model
    pipeline.train(
        model_name=model_name,
        model_dir=model_dir,
        dataset=dataset,
        dataset_extras=dataset_extras,
        dataset_format=dataset_format,
    )

    # End mlflow
    mlflow_end()

    logger.info(f"Fine tuning of model {model_name} done")
