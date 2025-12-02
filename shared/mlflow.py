import os
import time
from typing import Literal
import mlflow
import mlflow.data
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.data.huggingface_dataset import from_huggingface
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from datasets import Dataset
from shared.dataset import get_commit_hash, get_file
from shared.logger import get_logger

logger = get_logger(__name__)

RUN_TYPES = Literal["training", "inference", "evaluation", "testing"]
RUN_TYPES_TAGS = {
    "training": "train",
    "inference": "infer",
    "evaluation": "eval",
    "testing": "test",
}

def _sanitize_name(name: str) -> str:
    """Make sure the name fits alpha num naming rules (A-Za-z0-9_.-)"""
    last_name = name.split("/")[-1]
    return last_name.replace(" ", "-").replace(":", "-").lower()


def mlflow_enabled():
    if os.getenv("MLFLOW_TRACKING_URI"):
        return True
    return False


def mlflow_run_name(model_name: str, run_type: RUN_TYPES = None):
    run_name = _sanitize_name(model_name)
    custom_name = os.getenv("MLFLOW_RUN_NAME")
    run_tag = os.getenv("MLFLOW_RUN_NAME_TAG") or RUN_TYPES_TAGS.get(run_type or "")
    if custom_name:
        run_name += f"- {_sanitize_name(custom_name)}"
    if run_tag:
        run_name += f"-{run_tag}"
    return run_name


def mlflow_log_dataset(dataset_name: str, dataset: Dataset, dataset_split: str = "train", **metadata):
    if not mlflow_enabled():
        return

    name = _sanitize_name(dataset_name)
    commit_hash = get_commit_hash(dataset)
    if commit_hash:
        metadata["commit_hash"] = commit_hash
        dataset_source = HuggingFaceDatasetSource(dataset_name, split=dataset_split)
        mlflow_dataset = from_huggingface(dataset, source=dataset_source, name=name)
        mlflow.log_input(mlflow_dataset, context="training", tags=metadata)
    else:
        dataset_source = FileSystemDatasetSource(uri=f"s3://{get_file(dataset_name)}")
        mlflow_dataset = MetaDataset(source=dataset_source, name=name)
        mlflow.log_input(dataset, context="training", tags=metadata)


def mlflow_log_params(params: dict):
    if not mlflow_enabled():
        return

    for key, value in params.items():
        if len(str(value)) < 250:
            mlflow.log_param(key, value)
        else:
            mlflow.log_text(text=str(value), artifact_file=f"{key}.txt")

def mlflow_set_tags(tags: dict):
    if not mlflow_enabled():
        return
    
    mlflow.set_tags(tags)

def mlflow_log_artifact(local_path: str, artifact_path: str = None):
    if not mlflow_enabled():
        return

    mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)


def mlflow_log_model(model_name: str, model, tokenizer):
    if not mlflow_enabled():
        return

    model_name = os.getenv("MLFLOW_MODEL_NAME") or os.getenv("HF_PUSH_REPO") or model_name
    model_info = mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        tokenizer=tokenizer,
        name="model",
        registered_model_name=_sanitize_name(model_name),
    )
    mlflow.set_tags({"model_version": model_info.registered_model_version, "model_id": model_info.model_id})


def mlflow_active_model(model_name: str = None, model_id: str = None):
    if not mlflow_enabled():
        return

    model_id = model_id or os.getenv("MLFLOW_ACTIVE_MODEL_ID")
    model_name = model_name or os.getenv("MLFLOW_MODEL_NAME")
    if not model_id or not model_name:
        logger.warning(f"No model_id and model_name not found, traces won't be linked to a model!")
        return

    mlflow.set_active_model(model_id=model_id, model_name=model_name)


def mlflow_start(model_name: str, run_type: RUN_TYPES = None, tags: dict = None):
    if not mlflow_enabled():
        return

    if run_type:
        if tags:
            tags["run_type"] = run_type
        else:
            tags = {"run_type": run_type}

    # Look for env MLFLOW_EXPERIMENT_NAME else 'Default'
    mlflow.start_run(run_name=mlflow_run_name(model_name), tags=tags)


def mlflow_end():
    if not mlflow_enabled():
        return

    mlflow.end_run()

def mlflow_report_to():
    return "mlflow" if mlflow_enabled() else "none"
