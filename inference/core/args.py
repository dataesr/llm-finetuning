import argparse


def get_args():
    """
    Get script arguments

    Returns:
    - Namespace: arguments
    """
    parser = argparse.ArgumentParser(description="Run a vllm inference job")

    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model ID or path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset from HuggingFace or OVH file")
    parser.add_argument("--dataset_split", type=str, default=None, help="Dataset split")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config")

    return parser.parse_args()
