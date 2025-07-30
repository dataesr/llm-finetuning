import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from project.model.utils import model_get_finetuned_dir
from project.dataset import save_dataset_instruction, TEXT_FIELD, INSTRUCTION_FIELD
from project.logger import get_logger

logger = get_logger(__name__)

# LORA config (https://huggingface.co/docs/peft/package_reference/lora)
lora_r = 64  # Lora attention dimension (the “rank”).
lora_alpha = 16  # The alpha parameter for Lora scaling
lora_dropout = 0.1  # The dropout probability for Lora layers.
lora_task_type = "CAUSAL_LM"
lora_target_modules = ["q_proj", "v_proj"]
# For a deeper fine tuning use ["q_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]

# BitsAndBytesConfig int-4 config (https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/quantization#transformers.BitsAndBytesConfig)
device_map = {"": 0}
bnb_load_in_4bit = True  # Enable 4-bit quantization
bnb_4bit_use_double_quant = False  # Do not use nested quantization
bnb_4bit_compute_dtype = "float16"  # Computational type
bnb_4bit_quant_type = "nf4"  # Quantization data type

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
max_steps = 10  # Was originally trained on 3000 but better to keep the value low for test purposes.
# Context window length. Llama can now technically push further, but I find attention is no longer working so well.
max_seq_length = 8192
num_train_epochs = 1  # Number of training epochs
per_device_train_batch_size = 1  # Batch size per device during training. Optimal given our GPU vram.
gradient_accumulation_steps = 4  # Number of steps before performing a backward/update pass
gradient_checkpointing = True  # Use gradient checkpointing to save memory
optim = "paged_adamw_32bit"  # Use paged adamw optimizer
logging_steps = 10  # Log every 10 step
save_steps = 200  # We're going to save the model weights every 200 steps to save our checkpoint
learning_rate = 3e-4  # The initial learning rate for AdamW optimizer
fp16 = True  # Use fp16 precision
bf16 = False  # Do not use bfloat16 precision
max_grad_norm = 0.3  # Max gradient norm
warmup_ratio = 0.03  # Warmup ratio
lr_scheduler_type = "linear"  # Learning rate scheduler. Better to decrease the learning rate for long training. I prefer linear over to cosine as it is more predictable: easier to restart training if needed.
report_to = "none"  # Report metrics to tensorboard
group_by_length = True  # Group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient)
packing = False  # Pack short exemple to ecrease efficiency


def load_model_and_tokenizer(model_name: str):
    """
    Load pretrained causal model from huggingface

    Args:
    - model_name (str): Model to load

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

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set chat template to OAI chatML
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # model.resize_token_embeddings(len(tokenizer))
    logger.debug(f"Model embeddings size: {model.get_input_embeddings().weight.size(0)}")
    logger.debug(f"Tokenizer template: {tokenizer.chat_template}")
    logger.info(f"✅ Model and tokenizer loaded")

    return model, tokenizer

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
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
        packing=packing,
        max_seq_length=max_seq_length,
        # dataset_text_field="conversations" # automatically handled
    )

    # Build lora config
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        task_type=lora_task_type,
        target_modules=lora_target_modules,
    )

    # Build sft trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    return trainer


def construct_one_conversation(system: str, user: str, assistant: str = None):
    """
    Construct a conversation from system, user and assistant messages

    Args:
    - system (str): system instructions
    - user (str): user input
    - assistant (str, optional): assistant completion for training. Defaults to None.

    Returns a conversation object
    """
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if assistant:
        conversation.append({"role": "assistant", "content": assistant})
    return conversation


def construct_conversations(dataset: Dataset, completion_column: str) -> Dataset:
    """
    Construct conversations style column for training

    Args:
    - dataset (Dataset): training dataset
    - completion_column (str): completion column to use

    Returns the training dataset with a conversations column
    """

    def map_conversations(example):
        return {
            "conversations": construct_one_conversation(
                example[INSTRUCTION_FIELD],
                example["input"],
                example[completion_column],
            )
        }

    dataset = dataset.map(map_conversations)
    logger.debug(f"✅ Dataset formatted with conversation format")
    logger.debug(f"Dataset columns: {dataset.column_names}")
    logger.debug(f"Dataset conversations sample: {dataset[0]['conversations']}")
    return dataset


def merge_and_save_model(trainer, tokenizer, output_model_name: str, output_dir: str):
    """
    Save trained model and tokenizer.

    Args:
    - trainer: SFTTrainer
    - tokenizer: Tokenizer to save
    - output_model_name (str): Name of the saved model
    - output_dir (str): Path to output directory
    """
    logger.info(f"Start saving model to {output_dir}")

    # Get actual model object (handle DDP or not)
    model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

    # Save base checkpoint (with adapter if PEFT)
    model_to_save.save_pretrained(output_dir)
    logger.debug(f"Model embeddings size: {model_to_save.get_input_embeddings().weight.shape}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.debug(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load and merge model
    model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            device_map="auto",  
            torch_dtype=torch.bfloat16,
        )
    model = model.merge_and_unload()

    # Save final merged model
    output_merged_dir = model_get_finetuned_dir(output_model_name)
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)

    logger.info(f"✅ Fine-tuned model {output_model_name} merged and saved to {output_merged_dir}")

    # Cleanup
    torch.cuda.empty_cache()
    del model_to_save
    del model
    del trainer
    del tokenizer

def train(model_name: str, output_model_name: str, output_dir: str, dataset: Dataset, completion_column: str):
    """
    LLama model training pipeline

    Args:
        model_name (str): model to train
        output_model_name (str): model name to output
        output_dir (str): directory to output
        dataset (Dataset): training dataset
        completion_column (str): completion column to use in training dataset
    """
    logger.info(f"▶️ Start llama fine tuning pipeline")

    # Load the model and the tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Format dataset as conversations in new column
    dataset = construct_conversations(dataset, completion_column)

    # Train the model
    trainer = build_trainer(model, tokenizer, dataset, output_dir)
    trainer.train()
    logger.info("✅ Model trained")

    # Save the model
    merge_and_save_model(trainer, tokenizer, output_model_name, output_dir=output_dir)

    # Save the instruction
    save_dataset_instruction(dataset, destination=model_get_finetuned_dir(output_model_name))
