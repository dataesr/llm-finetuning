import os
import time
import mlflow
import mlflow.data
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.data.huggingface_dataset import from_huggingface
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from datasets import Dataset
from core.utils import model_get_finetuned_dir
from shared.dataset import get_commit_hash, get_file
from shared.logger import get_logger

logger = get_logger(__name__)

def _sanitize_name(name: str) -> str:
    """Make sure the name fits alpha num naming rules (A-Za-z0-9_.-)"""
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def mlflow_run_name(model_name: str):
    run_name = os.getenv("MLFLOW_RUN_NAME")
    if not run_name:
        run_name = f"{_sanitize_name(model_name)}-{os.getenv("MLFLOW_RUN_NAME_TAG") or _timestamp()}"
    return run_name


def mlflow_save_model(model, tokenizer, base_model_name: str, model_dir: str,  hf_hub: str = None, hf_hash: str = None, **metadata):
    run_id = mlflow.last_active_run().info.run_id
    artifact_name: str = mlflow_run_name(base_model_name)
    if hf_hub:
        artifact_name: str = _sanitize_name(hf_hub)
        metadata["model_source"] = "huggingface"
        metadata["hf_hub"] = hf_hub
        metadata["hf_hash"] = hf_hash
    else:
        metadata["model_source"] = "ovh"
        metadata["path"] = model_get_finetuned_dir(model_dir)
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("base_model_name", base_model_name)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            name=artifact_name,
            metadata=metadata
        )

def mlflow_log_dataset(dataset_name: str, dataset: Dataset, **metadata):
    name = _sanitize_name(dataset_name)
    commit_hash = get_commit_hash(dataset)
    if commit_hash:
        dataset_source = HuggingFaceDatasetSource(dataset_name)
        mlflow_dataset = from_huggingface(dataset, path=dataset_name, source=dataset_source, name=name)
        mlflow.log_input(mlflow_dataset, context="training", tags=metadata)
    else:
        dataset_source = FileSystemDatasetSource(uri=f"s3://{get_file(dataset_name)}")
        mlflow_dataset = MetaDataset(source=dataset_source, name=name)
        mlflow.log_input(dataset, context="training", tags=metadata)

def mlflow_log_prompts_params(params: dict):
    for key, value in params.items():
        mlflow.log_text(key, value)

def mlflow_log_lora_adapters(local_path: str):
    mlflow.log_artifact(local_path=local_path, artifact_path="lora_adapters")

def mlflow_log_model(model, tokenizer):
    mlflow.transformers.log_model(transformers_model=model, tokenizer=tokenizer, artifact_path="model")

def mlflow_start(model_name: str):
    # Look for env MLFLOW_EXPERIMENT_NAME
    mlflow.start_run(run_name=mlflow_run_name(model_name))

def mlflow_end():
    mlflow.end_run()
