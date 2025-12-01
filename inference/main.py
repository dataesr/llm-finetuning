"""
# Title : Fine-tuning
# Description : Fine-tuning script for @dataesr
"""

from core.args import get_args
from core.inference import inference
from shared.logger import get_logger

logger = get_logger(__name__)


def main():
    # Get script arguments
    args = get_args()

    ### Inference pipeline
    inference(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_config=args.dataset_config,
    )


if __name__ == "__main__":
    main()
