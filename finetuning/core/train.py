import importlib
from shared.mlflow import mlflow_start, mlflow_end, mlflow_log_dataset, mlflow_log_params
from shared.dataset import get_dataset
from shared.logger import get_logger

logger = get_logger(__name__)


def model_train(model_name: str, model_dir: str, pipeline_name: str, dataset_name: str, **kwargs) -> str:
    logger.info(f"🚀 Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Start mlflow
    mlflow_start(model_name, run_type="training", tags={"model_name": model_name, "dataset_name": dataset_name})

    # Load dataset
    dataset, dataset_extras = get_dataset(dataset_name, **kwargs)
    mlflow_log_dataset(dataset_name, dataset)
    mlflow_log_params(dataset_extras)

    # Get pipeline
    pipeline = importlib.import_module(f"core.pipeline.{pipeline_name}")

    # Train model
    pipeline.train(model_name=model_name, model_dir=model_dir, dataset=dataset, dataset_extras=dataset_extras)

    # End mlflow
    mlflow_end()

    logger.info(f"Fine tuning of model {model_name} done")
