"""
# Title : Fine-tuning
# Description : Fine-tuning script for @dataesr
"""

from core.args import get_args
from core.utils import model_delete_dir, model_initialize_dir
from core.train import model_train
from core.push import model_push_to_hub
from shared.logger import get_logger

logger = get_logger(__name__)


def main():
    # Get script arguments
    args = get_args()
    model_name = args.model_name
    script_mode = args.mode

    ### Fine-tuning pipeline
    if script_mode == "train":

        logger.debug(f"Start fine-tuning script with args {args}")

        # Initalize model folder
        _, model_dir = model_initialize_dir(model_name)

        # Start model training
        model_train(
            model_name=model_name,
            model_dir=model_dir,
            pipeline_name=args.pipeline,
            dataset_name=args.dataset_name,
            dataset_format=args.dataset_format,
            dataset_config=args.dataset_config,
        )

        # Push model to huggingface
        hf_hub = args.hf_hub
        hf_hash = None
        if hf_hub:
            hf_hash = model_push_to_hub(model_dir, hf_hub, args.hf_hub_private)
            if hf_hash:
                model_delete_dir(model_dir)

    ### Upload model to hub
    elif script_mode == "push":

        if not args.push_model_dir:
            raise ValueError("--push_model_dir must be specified in push mode")
        if not args.hf_hub:
            raise ValueError("--hf_hub must be specified in push mode")

        logger.debug(f"Start pushing model to hugging face hub with args {args}")

        model_push_to_hub(args.push_model_dir, args.hf_hub, args.hf_hub_private)
        model_delete_dir(args.push_model_dir)

    else:
        raise ValueError(f"Incorrect mode {args.mode}")


if __name__ == "__main__":
    main()
