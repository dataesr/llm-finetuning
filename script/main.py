"""
# Title : Fine-tuning
# Description : Fine-tuning script for @dataesr
"""

from args import get_args
from hugging import upload_model_to_hub
from pipeline import fine_tune, delete_model
from logger import get_logger

logger = get_logger(__name__)


def main():
    # Get script arguments
    args = get_args()

    # Fine-tuning pipeline
    if args.mode == "train":
        logger.debug(f"Start fine-tuning script with args {args}")
        fine_tune(args.model_name, args.dataset_name, args.output_model_name, args.hf_hub, args.hf_hub_private)

    # Upload model to hub
    elif args.mode == "push":
        if not args.output_model_name or not args.hf_hub:
            raise ValueError("Both --output_model_name and --hf_hub must be specified in push mode.")

        logger.debug(f"Start pushing model to hugging face hub with args {args}")
        upload_model_to_hub(args.output_model_name, args.hf_hub, args.hf_hub_private)

    # Delete model files
    elif args.mode == "delete":
        if not args.output_model_name:
            raise ValueError("--output_model_name must be specified in delete mode.")

        logger.debug(f"Start delete model file from {args.output_model_name}")
        delete_model(args.output_model_name)

if __name__ == "__main__":
    main()
