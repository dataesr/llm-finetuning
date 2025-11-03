import os
import time
import wandb
from typing import Literal
from datasets import Dataset
from core.utils import model_get_finetuned_dir
from shared.dataset import get_commit_hash, get_file


def _sanitize_name(name: str) -> str:
    """Make sure the name fits W&B naming rules (A-Za-z0-9_.-)"""
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def wandb_init(model_name: str):
    # add log checkpoints
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # login
    wandb.login(key=os.getenv("WANDB_KEY"))

    # init run
    run_name = f"{_sanitize_name(model_name)}-{os.getenv("WANDB_NAME") or _timestamp()}"
    wandb.init(name=run_name, job_type="train")


def wandb_add_artifact(name: str, type: Literal["dataset", "model"], **metadata):
    artifact = wandb.Artifact(name=name, type=type, metadata=metadata)
    wandb.run.log_artifact(artifact)


def wandb_add_dataset_artifact(dataset_name: str, dataset: Dataset, **metadata):
    artifact_name = {_sanitize_name(dataset_name)}
    commit_hash = get_commit_hash(dataset)
    if commit_hash:
        metadata["dataset_source"] = "huggingface"
        metadata["hf_hub"] = dataset_name
        metadata["hf_hash"] = commit_hash
    else:
        metadata["dataset_source"] = "ovh"
        metadata["path"] = get_file(dataset_name)

    wandb_add_artifact(name=artifact_name, type="dataset", **metadata)


def wandb_add_model_artifact(model_dir: str, hf_hub: str = None, hf_hash: str = None, **metadata):
    run_name = wandb.run.name
    artifact_name = run_name
    if hf_hub:
        artifact_name = _sanitize_name(hf_hub)
        metadata["model_source"] = "huggingface"
        metadata["hf_hub"] = hf_hub
        metadata["hf_hash"] = hf_hash
    else:
        metadata["model_source"] = "ovh"
        metadata["path"] = model_get_finetuned_dir(model_dir)
    wandb_add_artifact(name=artifact_name, type="model", **metadata)


def wandb_finish():
    wandb.run.finish()
