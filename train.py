"""
# Title : Fine-tuning
# Description : Fine-tuning script for @dataesr
"""

from project.args import get_args
from project.hugging import push_to_hub
from project.pipeline import fine_tune, delete_model
from project.logger import get_logger

logger = get_logger(__name__)


def main():
    # Get script arguments
    args = get_args()

    # Fine-tuning pipeline
    if args.mode == "train":
        logger.debug(f"Start fine-tuning script with args {args}")

        output_model_name = fine_tune(args.model_name, args.dataset_name, args.output_model_name, args.use_chatml)

        if args.hf_hub:
            push_to_hub(output_model_name, args.hf_hub, args.hf_hub_private)

        delete_model(output_model_name)

    # Upload model to hub
    elif args.mode == "push":

        if not args.output_model_name:
            raise ValueError("--output_model_name must be specified in push mode")
        if not args.hf_hub:
            raise ValueError("--hf_hub must be specified in push mode")

        logger.debug(f"Start pushing model to hugging face hub with args {args}")

        push_to_hub(args.output_model_name, args.hf_hub, args.hf_hub_private)

        delete_model(args.output_model_name)

    else:
        raise ValueError(f"Incorrect mode {args.mode}")


if __name__ == "__main__":
    main()
