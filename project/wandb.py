import os
import wandb


def wandb_init():
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init()


def wandb_add_config(config: dict):
    if config:
        wandb.config.update(config)
