import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from _utils import get_default_output_name, reset_folder
from dataset import get_dataset, TEXT_FIELD
from logger import get_logger

FOLDER = "jobs"
MERGED_FOLDER = "merged"

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unsloth config (https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-unsloth-notebook)
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# LORA config (https://huggingface.co/docs/peft/package_reference/lora)
lora_r = 64  # Lora attention dimension (the “rank”).
lora_alpha = 16  # The alpha parameter for Lora scaling
lora_dropout = 0  # The dropout probability for Lora layers.
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
max_steps = -1  # for a full run, and turn off max_steps=-1
max_seq_length = 8192

# Unsloth params (https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)
num_train_epochs = 1  # Number of training epochs
per_device_train_batch_size = 1  # Batch size per device during training. Optimal given our GPU vram.
gradient_accumulation_steps = 4  # Number of steps before performing a backward/update pass
optim = "adamw_8bit"  # Use adamw optimizer
logging_steps = 10  # Log every 10 step
learning_rate = 2e-4  # The initial learning rate for AdamW optimizer
fp16 = not is_bfloat16_supported()  # Use fp16 precision
bf16 = is_bfloat16_supported()  # Do not use bfloat16 precision
weight_decay = 0.01
warmup_steps = 5
lr_scheduler_type = "linear"  # Learning rate scheduler. Better to decrease the learning rate for long training. I prefer linear over to cosine as it is more predictable: easier to restart training if needed.
report_to = "none"  # Report metrics to something
group_by_length = True  # Group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient)
packing = False  # Pack short exemple to ecrease efficiency
seed = 1234

def initialize(model_name: str, output_model_name=None) -> tuple:
    """
    Initialize llm folder

    Args:
    - model_name (str): Base model to finetune
    - output_model_name (str): Finetuned model name. Default to None

    Returns:
    - output_model_name (str): Finetuned model name
    - output_dir (str): Finetuned model directory
    """
    # Default model name
    if not output_model_name:
        output_model_name = get_default_output_name(model_name)

    if not os.path.isdir(FOLDER):
        raise FileNotFoundError(f"Folder {FOLDER} not found on storage!")

    # Reset output folder
    output_dir = f"{FOLDER}/{output_model_name}"
    reset_folder(output_dir)

    return output_model_name, output_dir


def load_pretrained_model(model_name: str):
    """
    Load pretrained model from huggingface

    Args:
    - model_name (str): Model to load

    Returns:
    - model: Loaded model
    - tokenizer: Loaded tokenizer
    """
    logger.info(f"Start loading model {model_name}")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_length, dtype=dtype, load_in_4bit=load_in_4bit
    )

    # Add LoRa adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Set chat template to OAI chatML
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # model.resize_token_embeddings(len(tokenizer))
    logger.debug(f"Model embeddings size: {model.get_input_embeddings().weight.size(0)}")

    torch.cuda.empty_cache()

    logger.info(f"✅ Model and tokenizer loaded")

    return model, tokenizer


def train_model(model, tokenizer, dataset, output_dir: str):
    """
    Train model with custom dataset

    Args:
    - model: Model to be trained
    - tokenizer: Model tokenizer
    - dataset: Dataset to use for training
    - output_dir (str): Trained model directory

    Returns:
    - Trainer: Supervised Fine-Tuning trainer
    """
    logger.info(f"Start training model")

    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_steps=max_steps,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to,
        packing=packing,
        max_seq_length=max_seq_length,
        dataset_text_field=TEXT_FIELD,
        dataset_num_proc=2,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Training
    trainer_stats = trainer.train()

    logger.info("✅ Model trained")
    logger.debug(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.debug(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

    return trainer


def save_model(trainer, tokenizer, output_model_name: str, output_dir: str):
    """
    Save trained model

    Args:
    - trainer: SFT trainer
    - tokenizer: Model tokenizer
    - output_model_name (str): Trained model name
    - output_dir (str): Trained model directory
    """

    logger.info(f"Start saving model to {output_dir}")

    model_to_save = (
        trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained_merged(
        os.path.join(output_dir, MERGED_FOLDER), tokenizer, save_method="merged_16bit", safe_serialization=True
    )

    logger.debug(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.debug(f"Model embeddings size: {model_to_save.get_input_embeddings().weight.shape}")

    # Free memory
    torch.cuda.empty_cache()
    del model_to_save
    del tokenizer
    del trainer

    logger.info(f"✅ Fine-tuned model {output_model_name} saved")


def delete_model(output_model_name: str):
    """
    Delete all model files

    Args:
    - model_dir (str): model folder
    """
    model_dir = os.path.join(FOLDER, output_model_name)
    reset_folder(model_dir, delete=True)

    logger.info(f"✅ Model folder {model_dir} deleted")


def fine_tune(model_name: str, dataset_name: str, output_model_name=None):
    """
    Fine-tuning pipeline

    Args:
    - model_name (str): Model to fine-tune
    - dataset_name (str): Dataset to use for fine-tuning
    - output_model_name (str): Fine-tuned model name. Default to None
    - hub (str): huggingface hub to upload to. Default to None
    """

    logger.info(f"Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Initialize llm folder
    output_model_name, output_dir = initialize(model_name, output_model_name)

    # Load the model and the tokenizer
    model, tokenizer = load_pretrained_model(model_name)

    # Load dataset
    dataset = get_dataset(dataset_name, eos_token=tokenizer.eos_token)

    # Train the model
    trainer = train_model(model, tokenizer, dataset=dataset, output_dir=output_dir)

    # Save the mode
    save_model(trainer, tokenizer, output_dir=output_dir, output_model_name=output_model_name)

    return output_model_name
