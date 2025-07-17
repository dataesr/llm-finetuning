import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig

from project._utils import get_default_output_name, reset_folder
from project.dataset import get_dataset, save_dataset_instruction, TEXT_FIELD
from project.logger import get_logger

FOLDER = "job"
MERGED_FOLDER = "merged"

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train_model(
    model, tokenizer, dataset, sft_config: SFTConfig = None, peft_config: LoraConfig = None, data_collator: bool = None
):
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    # Training
    trainer.train()

    logger.info("✅ Model trained")

    return trainer


def save_model(trainer, output_model_name: str, output_dir: str, tokenizer=None, processor=None):
    """
    Save trained model (supports PEFT + non-PEFT) and tokenizer/processor.

    Args:
        trainer: SFTTrainer or Huggingface Trainer
        output_model_name (str): Name for the saved model
        output_dir (str): Path to output directory
        tokenizer: Optional tokenizer to save
        processor: Optional processor to save
    """
    logger.info(f"Start saving model to {output_dir}")

    # Get actual model object (handle DDP or not)
    model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

    # Save base checkpoint (with adapter if PEFT)
    model_to_save.save_pretrained(output_dir)
    logger.debug(f"Model embeddings size: {model_to_save.get_input_embeddings().weight.shape}")

    # Save tokenizer or processor
    if processor:
        processor.save_pretrained(output_dir)
        logger.debug(f"Tokenizer vocab size: {len(processor.tokenizer)}")
    elif tokenizer:
        tokenizer.save_pretrained(output_dir)
        logger.debug(f"Tokenizer vocab size: {len(tokenizer)}")

    # Check if model is PEFT (LoRA etc.)
    is_peft = isinstance(model_to_save, PeftModel)

    if is_peft:
        logger.info("PEFT model detected, merging adapters before saving final model...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            device_map="auto",  # Or customize device_map if needed
            torch_dtype=torch.bfloat16,  # Adjust dtype if necessary
        )
        model = model.merge_and_unload()
    else:
        model = model_to_save  # Already a full model

    # Save final merged model
    output_merged_dir = os.path.join(output_dir, MERGED_FOLDER)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    if processor:
        processor.save_pretrained(output_merged_dir)
    elif tokenizer:
        tokenizer.save_pretrained(output_merged_dir)

    logger.info(f"✅ Fine-tuned model {output_model_name} saved to {output_merged_dir}")

    # Cleanup
    torch.cuda.empty_cache()
    del model
    del trainer
    if tokenizer:
        del tokenizer
    if processor:
        del processor


def delete_model(output_model_name: str):
    """
    Delete all model files

    Args:
    - model_dir (str): model folder
    """
    model_dir = os.path.join(FOLDER, output_model_name)

    try:
        reset_folder(model_dir, delete=True)
        logger.info(f"✅ Model folder {model_dir} deleted")
    except Exception as error:
        logger.debug(f"Cannot delete folder {model_dir}: {error}")
