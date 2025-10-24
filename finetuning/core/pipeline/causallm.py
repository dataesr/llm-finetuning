import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from core.utils import model_get_output_dir, model_get_finetuned_dir
from shared.dataset import INSTRUCTION_FIELD, TEXT_FORMAT_FIELD, construct_prompts
from shared.utils import should_use_conversational_format
from shared.logger import get_logger

logger = get_logger(__name__)

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
NUM_TRAIN_EPOCHS = 3  # Number of training epochs
MAX_STEPS = 10  # 5_000  # Number of training steps, should be set to -1 for full training
BATCH_SIZE = 1  # Batch size per device during training. Optimal given our GPU vram.
GRAD_ACC_STEPS = 4  # Number of steps before performing a backward/update pass
OPTIM = "paged_adamw_8bit"
LEARNING_RATE = 2e-5  # The initial learning rate for AdamW optimizer
LR_SCHEDULER = "linear"  # Scheduler rate type
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
SAVE_STEPS = 500
LOG_STEPS = 1

# LORA config (https://huggingface.co/docs/peft/package_reference/lora)
LORA_R = 16
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1
TASK_TYPE = TaskType.CAUSAL_LM

# BitsAndBytesConfig int-4 config (https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/quantization#transformers.BitsAndBytesConfig)
BNB_4BIT = True
BNB_QUANT_TYPE = "nf4"
BNB_DOUBLE_QUANT = False
BNB_COMPUTE_DTYPE = torch.bfloat16


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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if not tokenizer:
        # Use fast if tokenizer not correctly loaded
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"  # to prevent warnings
    tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=BNB_4BIT,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for lora
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
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
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        task_type=TASK_TYPE,
        bias="none",
        # target_modules=["q_proj", "v_proj"],
        target_modules="all-linear",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # for full training
    )

    # Build sft config
    training_args = SFTConfig(
        output_dir=model_get_output_dir(model_dir),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        bf16=True,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIM,
        save_steps=SAVE_STEPS,
        logging_steps=LOG_STEPS,
        packing=False,
        report_to="wandb",
    )

    # Build sft trainer
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args, peft_config=lora_config
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

    # Check if it's actually a PEFT model
    if hasattr(model, "merge_and_unload"):
        # It's a PeftModel, save adapters
        model.save_pretrained(output_dir)
        # Reload model
        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
        # Merge model
        model_merged = model.merge_and_unload()
    else:
        # Fallback - just save the adapter weights
        logger.warning("⚠️ Could not merge PEFT weights, saving adapter only")

        # Save adapter weights
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"✅ Fine-tuned adapters saved to {output_dir}")
        torch.cuda.empty_cache()
        return

    # Save final merged model
    output_merged_dir = model_get_finetuned_dir(model_dir)
    model_merged.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)

    logger.info(f"✅ Fine-tuned model {model_dir} merged and saved to {output_merged_dir}")

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
    logger.info(f"▶️ Start causalLM finetuning pipeline")

    # Dataset custom prompts params
    custom_instruction = kwargs.get("dataset_extras", {}).get(INSTRUCTION_FIELD)
    custom_text_format = kwargs.get("dataset_extras", {}).get(TEXT_FORMAT_FIELD)

    # Load the model and the tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Format dataset for training
    use_conversational_format = should_use_conversational_format(kwargs.get("dataset_format"), tokenizer.chat_template)
    dataset = construct_prompts(
        dataset,
        custom_instruction=custom_instruction,
        custom_text_format=custom_text_format,
        use_conversational_format=use_conversational_format,
    )

    # Train the model
    output_dir = model_get_output_dir(model_dir)
    trainer = build_trainer(model, tokenizer, dataset, output_dir=output_dir)
    trainer.train()
    logger.info("✅ Model trained")

    # Save the model
    merge_and_save_model(trainer, tokenizer, output_dir=output_dir)
