from datasets import Dataset
from transformers.data.data_collator import DataCollatorForSeq2Seq
# unlosth has to be imported before trl see https://stackoverflow.com/questions/79663362/sfttrainer-the-specified-eos-token-eos-token-is-not-found-in-the-vocabu
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer
from core.utils import get_env, model_get_checkpoints_dir, model_get_finetuned_dir, model_get_output_dir
from shared.mlflow import mlflow_report_to, mlflow_log_model
from shared.dataset import INSTRUCTION_FIELD, TEXT_FORMAT_FIELD, construct_prompts
from shared.utils import should_use_conversational_format
import torch
from shared.logger import get_logger

logger = get_logger(__name__)

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
NUM_TRAIN_EPOCHS = get_env("NUM_TRAIN_EPOCHS", 3, int)  # Number of training epochs
MAX_STEPS = get_env("MAX_STEPS", -1, int)  # 5_000  # Number of training steps, should be set to -1 for full training
BATCH_SIZE = get_env("BATCH_SIZE", 2, int)  # Batch size per device during training. Optimal given our GPU vram.
GRAD_ACC_STEPS = get_env("GRAD_ACC_STEPS", 4, int)  # Number of steps before performing a backward/update pass
OPTIM = get_env("OPTIM", "adamw_8bit", str)
LEARNING_RATE = get_env("LEARNING_RATE", 2e-5, float)  # The initial learning rate for AdamW optimizer
LR_SCHEDULER = get_env("LR_SCHEDULER", "linear", str)  # Scheduler rate type
WEIGHT_DECAY = get_env("WEIGHT_DECAY", 0.001, float)
MAX_GRAD_NORM = get_env("MAX_GRAD_NORM", 0.3, float)
WARMUP_RATIO = get_env("WARMUP_RATIO", 0.03, float)
WARMUP_STEPS = get_env("WARMUP_STEPS", 5, int)
SAVE_STEPS = get_env("SAVE_STEPS", 500, int)
LOG_STEPS = get_env("LOG_STEPS", 1, int)

# LORA config (https://huggingface.co/docs/peft/package_reference/lora)
LORA_R = get_env("LORA_R", 16, int)
LORA_ALPHA = get_env("LORA_ALPHA", 16, int)
LORA_DROPOUT = get_env("LORA_DROPOUT", 0.0, float)

MAX_SEQ_LENGTH = get_env("MAX_SEQ_LENGHT", 8192, int)  # Choose any! We auto support RoPE Scaling internally!

dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",  # NEW! Llama 3.3 70B!
]  # More models at https://huggingface.co/unsloth


def is_unsloth_model(model_name: str, limit_to_4bit: bool = False):
    if limit_to_4bit:
        return model_name in fourbit_models
    return model_name.startswith("unsloth/")


def load_model_and_tokenizer(model_name: str):
    """
    Load pretrained causal model from huggingface

    Args:
    - model_name (str): Model to load
    - custom_chat_template: Custom chat template

    Returns:
    - model: Loaded model
    - tokenizer: Loaded tokenizer
    """
    logger.info(f"Start loading model {model_name}")

    if not is_unsloth_model(model_name, limit_to_4bit=True):
        raise ValueError(f"Model {model_name} is not a valid unsloth 4bit model!")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=69,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    logger.debug(f"Model embeddings size: {model.get_input_embeddings().weight.size(0)}")
    logger.debug(f"Tokenizer template: {tokenizer.chat_template}")
    logger.info(f"✅ Model and tokenizer loaded")

    return model, tokenizer


def build_trainer(model, tokenizer, dataset: Dataset, model_dir: str) -> SFTTrainer:
    """
    Build SFTTrainer for finetuning

    Args:
    - model: model to finetune
    - tokenizer: tokenizer:
    - dataset: Dataset
    - model_dir: model directory

    Returns:
    - trainer: SFTTrainer
    """

    # Build sft config
    training_args = SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,  # Set this for 1 full training run.
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOG_STEPS,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        bf16=False,  # doesnt work on non ampere gpu apparently
        fp16=True,
        seed=69,
        output_dir=model_get_checkpoints_dir(model_dir),
        report_to=mlflow_report_to(),
        max_length=MAX_SEQ_LENGTH,
        packing=False,  # Can make training 5x faster for short sequences.
    )

    # Build sft trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    )

    return trainer


def merge_and_save_model(trainer, tokenizer, model_dir: str):
    """
    Save trained model and tokenizer.

    Args:
    - trainer: trainer
    - tokenizer: tokenizer to save
    - model_dr (str): model directory
    """
    output_dir = model_get_output_dir(model_dir)
    logger.info(f"Start saving model to {output_dir}")

    # Get model from trainer
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

    # Save merged model as 16 bit
    output_merged_dir = model_get_finetuned_dir(model_dir)
    model.save_pretrained_merged(save_directory=output_merged_dir, tokenizer=tokenizer, save_method="merged_16bit")

    logger.info(f"✅ Fine-tuned model {model_dir} merged and saved to {output_merged_dir}")
    mlflow_log_model(model, tokenizer)

    # Cleanup
    torch.cuda.empty_cache()


def train(model_name: str, model_dir: str, dataset: Dataset, **kwargs):
    """
    Llama model training pipeline

    Args:
        model_name (str): model to train
        model_dir (str): model directory
        dataset (Dataset): training dataset
    """
    logger.info(f"▶️ Start unsloth-causallm finetuning pipeline")

    # Dataset custom prompts params
    dataset_extras = kwargs.get("dataset_extras") or {}
    dataset_format = dataset_extras.get("dataset_format")
    custom_instruction = dataset_extras.get(INSTRUCTION_FIELD)
    custom_text_format = dataset_extras.get(TEXT_FORMAT_FIELD)
    instruction_part = dataset_extras.get("instruction_part")
    response_part = dataset_extras.get("response_part")

    # Load the model and the tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Format dataset for training
    use_conversational_format = should_use_conversational_format(dataset_format, tokenizer.chat_template)
    dataset = construct_prompts(
        dataset,
        custom_instruction=custom_instruction,
        custom_text_format=custom_text_format,
        use_conversational_format=use_conversational_format,
    )

    # Train the model
    trainer = build_trainer(model, tokenizer, dataset, model_dir=model_dir)
    if instruction_part and response_part:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
        logger.info("Train model on responses only enabled!")
    trainer.train()
    logger.info("✅ Model trained")

    # Save the model
    merge_and_save_model(trainer, tokenizer, model_dir=model_dir)
