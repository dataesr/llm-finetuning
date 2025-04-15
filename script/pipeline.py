import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer, setup_chat_format
from code import dataset as datasets_service
from script.logging import get_logger

BUCKET = "llm-outputs"
FOLDER = "llm"

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LORA config (https://huggingface.co/docs/peft/package_reference/lora)
lora_r = 64  # Lora attention dimension (the “rank”).
lora_alpha = 16  # The alpha parameter for Lora scaling
lora_dropout = 0.1  # The dropout probability for Lora layers.
lora_task_type = ("CAUSAL_LM",)
lora_target_modules = (["q_proj", "v_proj"],)
# For a deeper fine tuning use ["q_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]

# BitsAndBytesConfig int-4 config (https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/quantization#transformers.BitsAndBytesConfig)
device_map = {"": 0}
bnb_load_in_4bit = True  # Enable 4-bit quantization
bnb_4bit_use_double_quant = False  # Do not use nested quantization
bnb_4bit_compute_dtype = "float16"  # Computational type
bnb_4bit_quant_type = "nf4"  # Quantization data type

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
max_steps = 600  # Was originally trained on 3000 but better to keep the value low for test purposes.
max_seq_length = (
    8192  # Context window length. Llama can now technically push further, but I find attention is no longer working so well.
)
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


def initialize(output_model_name: str) -> str:
    """Initialize llm folder"""
    is_folder = os.path.isdir(FOLDER)
    if not is_folder:
        logger.error(f"Folder {FOLDER} not found on storage!")

    output_dir = f"{FOLDER}/{output_model_name}"
    return output_dir


def load_pretrained_model(model_name: str):
    logger.debug(f"Start loading model {model_name}")

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"  # to prevent warinings
    tokenizer.pad_token = tokenizer.eos_token

    # Set chat template to OAI chatML
    model, tokenizer = setup_chat_format(model, tokenizer)

    torch.cuda.empty_cache()

    logger.debug(f"Model and tokenizer loaded!")

    return model, tokenizer


def train_model(model, tokenizer, dataset, output_dir: str):
    logger.debug(f"Start training model")

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        task_type=lora_task_type,
        target_modules=lora_target_modules,
    )

    training_arguments = TrainingArguments(
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
        # max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        packing=packing,
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
    )

    # Training
    trainer.train()

    logger.debug("Model trained!")

    return trainer


def save_model(trainer, tokenizer, output_model_name, output_dir, hub):

    logger.debug(f"Start saving model to {output_dir}/{output_model_name}")

    model_to_save = (
        trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_model_name)

    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_model_name, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(output_dir, output_model_name)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # We also save the tokenizer
    tokenizer.save_pretrained(output_merged_dir)

    # Push to hub if defined
    if hub:
        logger.debug(f"Pushing model and tokenizer to huggingface hub {hub}")
        model.push_to_hub(repo_id=hub)
        tokenizer.push_to_hub(repo_id=hub)

    # Free memory
    del model
    del tokenizer
    del trainer

    return True


def fine_tune(model_name: str, dataset_name: str, output_model_name: str, hub: str):
    logger.debug(f"Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Initialize llm folder
    output_dir = initialize(output_model_name)

    # Load dataset
    dataset = datasets_service.load(dataset_name)

    # Load the model and the tokenizer
    model, tokenizer = load_pretrained_model(model_name)

    # Train the model
    trainer = train_model(model, tokenizer, dataset=dataset, output_dir=output_dir)

    # Save the mode
    is_saved = save_model(trainer, tokenizer, output_dir=output_dir, output_model_name=output_model_name, hub=hub)

    if is_saved:
        logger.error("Error while saving model.")
    else:
        logger.error(f"New model {output_model_name} saved!")
