"""
# Title : Fine-tuning
# Description : Fine-tuning script for @dataesr
"""

from script.args import get_args
from script.pipeline import fine_tune
from script.logging import get_logger

logger = get_logger(__name__)


def main():
    # Get script arguments
    args = get_args()
    logger.debug(f"Start fine-tuning script with args {args}")

    # Fine-tuning pipeline
    # fine_tune(args.model_name, args.dataset_name, args.output_model_name, args.hf_hub)


if __name__ == "__main__":
    main()
