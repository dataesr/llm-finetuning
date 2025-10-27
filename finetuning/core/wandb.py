import os
import wandb
from typing import Literal


def wandb_init(name: str = None):
    # add log checkpoints
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # login
    wandb.login(key=os.getenv("WANDB_KEY"))

    # init run
    run_name = os.getenv("WANDB_NAME") or name
    wandb.init(name=run_name, job_type="train")


def wandb_add_artifact(name: str, type: Literal["dataset", "model"], **metadata):
    artifact = wandb.Artifact(name=name, type=type, metadata=metadata)
    wandb.run.use_artifact(artifact)


def wandb_finish():
    wandb.run.finish()
