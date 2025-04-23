import argparse


def get_args():
    """
    Get script arguments

    Returns:
    - Namespace: arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained model on a custom dataset")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "push", "delete"],
        default="train",
        help="Execution mode: train model, push model to hub, or delete model files",
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Base model to fine-tune")
    parser.add_argument("--dataset_name", type=str, default="test.json", help="Dataset to use for fine-tuning")
    parser.add_argument("--output_model_name", type=str, default=None, help="Fine-tuned model name")
    parser.add_argument("--hf_hub", type=str, default=None, help="Push the model to a Huggin Face Hub")
    parser.add_argument("--hf_hub_private", action="store_true", help="If set the hugging face hub will be private")

    return parser.parse_args()
