"""
# Title : Fine-tuning
# Description : Fine-tuning script for @dataesr
"""

from core.args import get_args
from core.wandb import wandb_init
from core.utils import model_delete_dir
from core.train import model_train
from core.push import model_push_to_hub
from shared.logger import get_logger

logger = get_logger(__name__)


def main():
    # Get script arguments
    args = get_args()

    # Fine-tuning pipeline
    if args.mode == "train":
        logger.debug(f"Start fine-tuning script with args {args}")

        # init wandb
        wandb_init()

        # start model training
        output_model_name = model_train(
            model_name=args.model_name,
            pipeline_name=args.pipeline,
            dataset_name=args.dataset_name,
            dataset_format=args.dataset_format,
            output_model_name=args.output_model_name,
        )

        # push to huggingface
        if args.hf_hub:
            model_push_to_hub(output_model_name, args.hf_hub, args.hf_hub_private)
            model_delete_dir(output_model_name)

    # Upload model to hub
    elif args.mode == "push":

        if not args.output_model_name:
            raise ValueError("--output_model_name must be specified in push mode")
        if not args.hf_hub:
            raise ValueError("--hf_hub must be specified in push mode")

        logger.debug(f"Start pushing model to hugging face hub with args {args}")

        model_push_to_hub(args.output_model_name, args.hf_hub, args.hf_hub_private)
        model_delete_dir(args.output_model_name)

    else:
        raise ValueError(f"Incorrect mode {args.mode}")


if __name__ == "__main__":
    main()
