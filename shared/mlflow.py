import os
import time
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

def _sanitize_name(name: str) -> str:
    """Make sure the name fits alpha num naming rules (A-Za-z0-9_.-)"""
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def mlflow_enabled():
    if os.getenv("MLFLOW_TRACKING_URI"):
        return True
    return False

def mlflow_run_name(model_name: str):
    run_name = os.getenv("MLFLOW_RUN_NAME")
    if not run_name:
        run_name = f"{_sanitize_name(model_name)}-{os.getenv("MLFLOW_RUN_NAME_TAG") or _timestamp()}"
    return run_name

def mlflow_log_dataset(dataset_name: str, dataset: Dataset, **metadata):
    if not mlflow_enabled():
        return
    
    name = _sanitize_name(dataset_name)
    commit_hash = get_commit_hash(dataset)
    if commit_hash:
        dataset_source = HuggingFaceDatasetSource(dataset_name)
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
    
def mlflow_log_lora_adapters(local_path: str):
    if not mlflow_enabled():
        return

    mlflow.log_artifact(local_path=local_path, artifact_path="lora_adapters")

def mlflow_log_model(model, tokenizer):
    if not mlflow_enabled():
        return

    model_name = os.getenv("MLFLOW_MODEL_NAME") or os.getenv("HF_PUSH_REPO") or "model"
    mlflow.transformers.log_model(transformers_model={"model": model, "tokenizer": tokenizer}, tokenizer=tokenizer, name=model_name)

def mlflow_active_model(model_id: str = None):
    if not mlflow_enabled():
        return
    
    model_id = os.getenv("MLFLOW_MODEL_ID") or model_id
    if not model_id:
        logger.warning(f"No model_id found, traces won't be linked to a model!")
        return

    mlflow.set_active_model(model_id=model_id)

def mlflow_start(model_name: str):
    if not mlflow_enabled():
        return

    # Look for env MLFLOW_EXPERIMENT_NAME
    mlflow.start_run(run_name=mlflow_run_name(model_name))

def mlflow_end():
    if not mlflow_enabled():
        return

    mlflow.end_run()
    
def mlflow_report_to():
    return "mlflow" if mlflow_enabled() else "none"
