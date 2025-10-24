import os
import importlib
import torch
from core.wandb import wandb_add_artifact
from shared.dataset import get_dataset, get_dataset_extras
from shared.logger import get_logger

logger = get_logger(__name__)


def model_train(model_name: str, model_output_name: str, pipeline_name: str, dataset_name: str, **kwargs) -> str:
    logger.info(f"🚀 Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Cleanup
    torch.cuda.empty_cache()

    # Load dataset
    dataset = get_dataset(dataset_name)
    dataset_extras = get_dataset_extras(dataset_name)
    wandb_add_artifact(
        name=dataset_name,
        type="dataset",
        dataset_len=len(dataset),
        dataset_features=dataset.features,
        **dataset_extras,
    )

    # Get pipeline
    pipeline = importlib.import_module(f"core.pipeline.{pipeline_name}")

    # Train model
    pipeline.train(
        model_name=model_name,
        model_dir=model_output_name,
        dataset=dataset,
        dataset_extras=dataset_extras,
        dataset_format=kwargs.get("dataset_format"),
    )

    logger.info(f"Fine tuning of model {model_name} done")
