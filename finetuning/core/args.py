import argparse


def get_args():
    """
    Get script arguments

    Returns:
    - Namespace: arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained model on a custom dataset")

    # Execution mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "push"],
        default="train",
        help="Execution mode: train model or push model to hub",
    )
    parser.add_argument("--push_model_dir", type=str, default=None, help="Fine-tuned model directory to push")

    # Model name
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Base model to fine-tune")

    # Finetuning pipeline
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["causallm", "causallm-unsloth"],
        default="causallm",
        help="Force a finetuning pipeline (eg llama (causallm), custom, ..)",
    )

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="test.json", help="Dataset to use for fine-tuning")
    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["auto", "text", "conversational"],
        help="How to format the dataset, either 'auto', 'text' or 'conversational'",
    )
    parser.add_argument(
        "--dataset_extras_name",
        type=str,
        default=None,
        help="Extras dataset params file name (should be store on llm-datasets/extras)",
    )

    # HuggingFace args
    parser.add_argument("--hf_hub", type=str, default=None, help="Push the model to a Huggin Face Hub")
    parser.add_argument("--hf_hub_private", action="store_true", help="If set, the hugging face hub will be private")

    return parser.parse_args()
