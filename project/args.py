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
        choices=["train", "push"],
        default="train",
        help="Execution mode: train model or push model to hub",
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Base model to fine-tune")
    parser.add_argument("--dataset_name", type=str, default="test.json", help="Dataset to use for fine-tuning")
    parser.add_argument("--output_model_name", type=str, default=None, help="Fine-tuned model name")
    parser.add_argument("--forced_config", type=str, default=None, help="Force a pipeline config (eg llama, qwen2_vl)")
    parser.add_argument("--hf_hub", type=str, default=None, help="Push the model to a Huggin Face Hub")
    parser.add_argument("--hf_hub_private", action="store_true", help="If set, the hugging face hub will be private")
    # parser.add_argument("--use_chatml", action="store_true", help="If set, use chatml tokenizer")

    return parser.parse_args()
