import mlflow
from typing import Dict, Any
from vllm import LLM
from vllm import LLM, SamplingParams
from vllm.version import __version__ as VLLM_VERSION
from vllm.transformers_utils.tokenizer import get_tokenizer
from core.prompts import apply_chat_template
from shared.mlflow import mlflow_log_params
from shared.logger import get_logger

logger = get_logger(__name__)


def load_engine(model_name):
    # Load tokenizer
    tokenizer = get_tokenizer(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    logger.info(f"✅ {model_name} tokenizer loaded")

    # Load vllm engine
    engine = LLM(
        model=model_name,
        quantization="bitsandbytes",
        dtype="bfloat16",
        tensor_parallel_size=1,  # torch.cuda.device_count()
        trust_remote_code=True,
        # enforce_eager=True,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
        max_model_len=8192,
    )
    logger.info(f"✅ vLLM engine {VLLM_VERSION} loaded")

    return engine, tokenizer


@mlflow.trace(name="vllm_generate", span_type="llm")
def generate(
    model_name: str, prompts: list[str], prompts_params: Dict[str, Any], sampling_params: Dict[str, Any]
) -> list[str]:
    # Load vllm engine and tokenizer
    engine, tokenizer = load_engine(model_name)

    logger.debug(f"Running generation on {len(prompts)} prompts...")
    if sampling_params:
        logger.debug(f"Custom sampling params: {sampling_params}")

    # Format prompts
    formatted_prompts = apply_chat_template(tokenizer, prompts, prompts_params=prompts_params)

    # Sampling params
    max_length = tokenizer.model_max_length
    truncate_length = max_length if isinstance(max_length, int) and max_length < 1_000_000 else 8192
    full_params = {
        "seed": 0,
        "temperature": 0,
        "max_tokens": max_length,
        "skip_special_tokens": True,
        "truncate_prompt_tokens": truncate_length,
        **sampling_params,
    }
    mlflow_log_params(full_params)

    # Generate outputs
    outputs = engine.generate(formatted_prompts, SamplingParams(**full_params))
    completions = [output.outputs[0].text for output in outputs]
    logger.debug(f"Generated {len(completions)} completions")

    return completions
