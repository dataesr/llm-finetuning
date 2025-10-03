import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from project.model.utils import model_get_finetuned_dir
from project.dataset import (
    save_dataset_extras,
    INSTRUCTION_FIELD,
    INPUT_FIELD,
    COMPLETION_FIELD,
    CONVERSATIONS_FIELD,
    CHAT_TEMPLATE_FIELD,
    TEXT_FIELD,
    TEXT_FORMAT_FIELD,
)
from project.logger import get_logger

logger = get_logger(__name__)

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
# https://github.com/numindai/nuextract/blob/main/cookbooks/nuextract-2.0_sft.ipynb
num_train_epochs = 3  # Number of training epochs
max_steps = -1  # Number of training steps, should be set to -1 for full training
per_device_train_batch_size = 1  # Batch size per device during training. Optimal given our GPU vram.
gradient_accumulation_steps = 4  # Number of steps before performing a backward/update pass
optim = "paged_adamw_8bit"
learning_rate = 2e-5  # The initial learning rate for AdamW optimizer
lr_scheduler_type = "linear"  # Scheduler rate type
max_seq_length = 8192  # Context window length. Llama can now technically push further

# BitsAndBytesConfig int-4 config (https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/quantization#transformers.BitsAndBytesConfig)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LORA config (https://huggingface.co/docs/peft/package_reference/lora)
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    # target_modules=["q_proj", "v_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # for full training
)

default_text_format = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"

def load_model_and_tokenizer(model_name: str, custom_chat_template=None):
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

    if custom_chat_template:
        tokenizer.chat_template = custom_chat_template
        logger.debug(f"Custom chat template: {tokenizer.chat_template}")
        logger.info("✅ Applied custom chat template")

    # Load model
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


def construct_one_conversation(system: str, user: str, assistant: str = None):
    """
    Construct a conversation from system, user and assistant messages

    Args:
    - system (str): system instructions
    - user (str): user input
    - assistant (str, optional): assistant completion for training. Defaults to None.

    Returns a conversation object
    """
    conversation = []

    # Add system prompt
    if system:
        conversation.append({"role": "system", "content": system})

    # Add user prompt
    conversation.append({"role": "user", "content": user})

    # Add assistant prompt
    if assistant:
        conversation.append({"role": "assistant", "content": assistant})

    return conversation


def construct_prompts(
    dataset: Dataset,
    custom_instruction: str = None,
    custom_text_format: str = None,
    use_conversational_format: bool = False,
) -> Dataset:
    """
    Construct prompts for training on a dataset

    Args:
    - dataset (Dataset): training dataset
    - custom_instruction (str): custom system prompt
    - custom_text_format (str): custom text format
    - use_conversational_format (bool): if True, use conversational format

    Returns the training dataset with a conversations column
    """
    prompts_field = CONVERSATIONS_FIELD if use_conversational_format else TEXT_FIELD

    def map_conversations(example):
        if use_conversational_format:
            # Conversational format (list of messages, ChatML-like)
            return {
                prompts_field: construct_one_conversation(
                    system=custom_instruction,
                    user=example[INPUT_FIELD],
                    assistant=example[COMPLETION_FIELD],
                )
            }
        else:
            # Non-conversational (Alpaca-style prompt-response text)
            instruction = custom_instruction or "You are an helpful assistant."
            text_format = custom_text_format if custom_text_format else default_text_format
            text = text_format.format(instruction, example[INPUT_FIELD], example[COMPLETION_FIELD])
            return {prompts_field: text}

    dataset = dataset.map(map_conversations).select_columns([prompts_field])
    logger.debug(f"✅ Dataset formatted with {'conversation' if use_conversational_format else 'text'} format")
    logger.debug(f"Dataset columns: {dataset.column_names}")
    logger.debug(f"Dataset sample: {dataset[0][prompts_field]}")
    return dataset


def build_trainer(model, tokenizer, dataset: Dataset, output_dir: str) -> SFTTrainer:
    """
    Build SFTTrainer for finetuning

    Args:
    - model: model to finetune
    - tokenizer: tokenizer:
    - dataset: Dataset
    - output_dir: training output directory

    Returns:
    - trainer: SFTTrainer
    """
    # Build sft config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        weight_decay=0.001,
        optim=optim,
        save_steps=200,
        logging_steps=10,
        logging_strategy="steps",
        logging_first_step=True,
        disable_tqdm=False,
        report_to=None,
        dataloader_pin_memory=False,
        # max_seq_length=max_seq_length,
        packing=False,
    )

    # Build sft trainer
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args, peft_config=lora_config
    )

    return trainer


def merge_and_save_model(trainer, tokenizer, output_model_name: str, output_dir: str):
    """
    Save trained model and tokenizer.

    Args:
    - trainer: trainer
    - tokenizer: tokenizer to save
    - output_model_name (str): Name of the saved model
    - output_dir (str): Path to output directory
    """
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
    output_merged_dir = model_get_finetuned_dir(output_model_name)
    model_merged.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)

    logger.info(f"✅ Fine-tuned model {output_model_name} merged and saved to {output_merged_dir}")

    # Cleanup
    torch.cuda.empty_cache()


def train(model_name: str, output_model_name: str, output_dir: str, dataset: Dataset, **kwargs):
    """
    Llama model training pipeline

    Args:
        model_name (str): model to train
        output_model_name (str): model name to output
        output_dir (str): directory to output
        dataset (Dataset): training dataset
    """
    logger.info(f"▶️ Start Llama fine tuning pipeline")

    # Dataset custom prompts params
    custom_instruction = kwargs.get("dataset_extras", {}).get(INSTRUCTION_FIELD)
    custom_text_format = kwargs.get("dataset_extras", {}).get(TEXT_FORMAT_FIELD)
    custom_chat_template = kwargs.get("dataset_extras", {}).get(CHAT_TEMPLATE_FIELD)

    # Load the model and the tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, custom_chat_template=custom_chat_template)
    if kwargs.get("no_chat_template"):
        tokenizer.chat_template = None
    use_conversational_format = tokenizer.chat_template is not None  # use conversational format for instruct models

    # Format dataset for training
    dataset = construct_prompts(
        dataset,
        custom_instruction=custom_instruction,
        custom_text_format=custom_text_format,
        use_conversational_format=use_conversational_format,
    )

    # Train the model
    trainer = build_trainer(model, tokenizer, dataset, output_dir)
    trainer.train()
    logger.info("✅ Model trained")

    # Save the model
    merge_and_save_model(trainer, tokenizer, output_model_name, output_dir=output_dir)

    # Save the extras
    save_dataset_extras(dataset, destination=model_get_finetuned_dir(output_model_name))
