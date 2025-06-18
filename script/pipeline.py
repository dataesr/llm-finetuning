import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig
from script._utils import get_default_output_name, reset_folder
from script.dataset import get_dataset, TEXT_FIELD
from script.logger import get_logger

FOLDER = "jobs"
MERGED_FOLDER = "merged"

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
max_steps = 600  # Was originally trained on 3000 but better to keep the value low for test purposes.
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
report_to = "tensorboard"  # Report metrics to tensorboard
group_by_length = True  # Group together samples of roughly the same length in the training dataset (to minimize padding applied and be more efficient)
packing = False  # Pack short exemple to ecrease efficiency


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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=bnb_config)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.padding_side = "right"  # to prevent warnings
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

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

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        task_type=lora_task_type,
        target_modules=lora_target_modules,
    )

    training_arguments = SFTConfig(
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
        dataset_text_field=TEXT_FIELD,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Training
    trainer.train()

    logger.info("✅ Model trained")

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
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.debug(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.debug(f"Model embeddings size: {model_to_save.get_input_embeddings().weight.shape}")

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map=device_map, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(output_dir, MERGED_FOLDER)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # We also save the tokenizer
    tokenizer.save_pretrained(output_merged_dir)

    # Free memory
    torch.cuda.empty_cache()
    del model
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


def fine_tune(model_name: str, dataset_name: str, output_model_name: str = None, use_chatml: bool = False):
    """
    Fine-tuning pipeline

    Args:
    - model_name (str): Model to fine-tune
    - dataset_name (str): Dataset to use for fine-tuning
    - output_model_name (str): Fine-tuned model name. Default to None
    - use_chatml (bool): if True, use chatml tokenizer
    """

    logger.info(f"Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Initialize llm folder
    output_model_name, output_dir = initialize(model_name, output_model_name)

    # Load the model and the tokenizer
    model, tokenizer = load_pretrained_model(model_name)

    # Load dataset
    dataset = get_dataset(dataset_name, tokenizer=tokenizer, use_chatml=use_chatml)

    # Train the model
    trainer = train_model(model, tokenizer, dataset=dataset, output_dir=output_dir)

    # Save the mode
    save_model(trainer, tokenizer, output_dir=output_dir, output_model_name=output_model_name)

    return output_model_name


def predict(model, tokenizer, input: str | object, use_chatml: bool) -> str:
    """
    Generate model prediction

    Args:
        model: LLM model
        tokenizer: LLM tokenizer
        input (str | object): input plain text or chatml object
        use_chatml (bool): use chat ml format

    Returns:
        prediction (str): generated prediction
    """
    logger.info(f"Start predict with input={input}")

    model.to(device)

    # Get input as text
    input_text = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True) if use_chatml else input

    # Get tensors
    input_tensors = tokenizer(input_text, padding=True, return_attention_mask=True, return_tesnros="pt").to(model.device)

    # Get outputs
    outputs = model.generate(**input_tensors, max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id)

    # Decode outputs
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return prediction
