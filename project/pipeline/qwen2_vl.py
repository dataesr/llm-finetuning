import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from project.model.utils import model_get_finetuned_dir
from project.dataset import save_dataset_instruction, INSTRUCTION_FIELD, TEXT_FIELD
from project.logger import get_logger

logger = get_logger(__name__)

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
# https://github.com/numindai/nuextract/blob/main/cookbooks/nuextract-2.0_sft.ipynb
num_train_epochs = 5  # Number of training epochs
per_device_train_batch_size = 1  # Batch size per device during training. Optimal given our GPU vram.
gradient_accumulation_steps = 4  # Number of steps before performing a backward/update pass
gradient_checkpointing = True  # Use gradient checkpointing to save memory
gradient_checkpointing_kwargs = {"use_reentrant": False}  # Options for gradient checkpointing
learning_rate = 1e-5  # The initial learning rate for AdamW optimizer
lr_scheduler_type = "constant"  # Scheduler rate type
bf16 = True  # Use bfloat16 precision
max_grad_norm = 0.3  # Max gradient norm
warmup_ratio = 0.03  # Warmup ratio
report_to = "none"  # Report metrics to tensorboard
logging_steps = 10  # Log every 10 step
save_steps = 200  # We're going to save the model weights every 200 steps to save our checkpoint


def load_model_and_processor(model_name: str):
    """
    Load model and processor

    Args:
    - model_name (str): Model to load

    Returns:
    - model: Loaded model
    - processor: Loaded processor
    """

    logger.info(f"Start loading model {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
        use_cache=False,  # for training
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",  # make sure to set padding to right for training
        use_fast=True,
    )
    processor.eos_token = processor.tokenizer.eos_token
    processor.eos_token_id = processor.tokenizer.eos_token_id

    logger.debug(f"Model embeddings size: {model.get_input_embeddings().weight.size(0)}")
    logger.debug(f"Tokenizer template: {processor.tokenizer.chat_template}")
    logger.info(f"✅ Model and tokenizer loaded")

    return model, processor


def construct_one_conversation(system: str, user: str, template: str, assistant: str = None):
    """
    Construct a conversation from system, user and assistant messages

    Args:
    - system (str): system instructions
    - user (str): user input
    - template (str): extraction template (for NuExtract)
    - assistant (str, optional): assistant completion for training. Defaults to None.

    Returns a conversation object
    """
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "text", "text": f"# Template:\n{template}\n# Context:\n{user}"}]},
    ]
    if assistant:
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": f"{assistant}"}]})
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
                example["template"],
                example[completion_column],
            )
        }

    dataset = dataset.map(map_conversations)
    logger.debug(f"✅ Dataset formatted with conversation format")
    logger.debug(f"Dataset columns: {dataset.column_names}")
    logger.debug(f"Dataset conversations sample: {dataset[0]['conversations']}")
    return dataset


# def build_collate_fn(processor):
#     """
#     Build data collator function.
#     Mask system and user prompts to reduce loss on assistant prompt only

#     Args: processor
#     Returns: inputs_ids, labels
#     """

#     def collate_fn(inputs):
#         # process system and user part of conversations
#         logger.debug(f"💥INPUTS: {inputs}")
#         assistant_idx = next(i for i, m in enumerate(inputs) if m["role"] == "assistant")
#         user_texts = [processor.apply_chat_template(input[:assistant_idx], tokenize=False) for input in inputs]

#         # process full conversations (system + user + assistant)
#         full_texts = [processor.apply_chat_template(input, tokenize=False) for input in inputs]

#         # process images
#         images = None

#         # tokenize sequences
#         user_batch = processor(text=user_texts, images=images, return_tensors="pt", padding=True)
#         full_batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)

#         # mask padding tokens
#         labels = full_batch["input_ids"].clone()
#         labels[labels == processor.tokenizer.pad_token_id] = -100

#         # mask user message tokens for each example in the batch
#         for i in range(len(inputs)):
#             # length of prompt message (accounting for possible padding)
#             user_len = user_batch["attention_mask"][i].sum().item()

#             # mask prompt part of label
#             labels[i, : user_len - 1] = -100

#         full_batch["labels"] = labels
#         return full_batch

#     return collate_fn


def build_trainer(model, processor, dataset: Dataset, output_dir: str) -> SFTTrainer:
    """
    Build SFTTrainer for finetuning

    Args:
    - model: model to finetune
    - processor: processor
    - dataset: Dataset
    - output_dir: training output directory

    Returns:
    - trainer: SFTTrainer
    """
    # Build collator fnc
    # data_collator = build_collate_fn(processor)

    # Build sft config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        # max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        # max_seq_length=max_seq_length,
    )

    # Build sft trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        args=training_args,
        # data_collator=data_collator,
    )

    return trainer


def save_model(trainer, processor, output_model_name: str, output_dir: str):
    """
    Save trained model and processor.

    Args:
        trainer: SFTTrainer
        processor: processor to save
        output_model_name (str): Name for the saved model
        output_dir (str): Path to output directory
    """
    model_dir = model_get_finetuned_dir(output_model_name)
    logger.info(f"Start saving model to {model_dir}")

    # Get actual model object (handle DDP or not)
    model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

    # Save model and tokenizer
    model_to_save.save_pretrained(model_dir)
    logger.debug(f"Model embeddings size: {model_to_save.get_input_embeddings().weight.shape}")

    # Save tokenizer or processor
    processor.save_pretrained(model_dir)
    logger.debug(f"Tokenizer vocab size: {len(processor.tokenizer)}")

    logger.info(f"✅ Fine-tuned model {output_model_name} saved to {model_dir}")

    # Cleanup
    torch.cuda.empty_cache()
    del model_to_save
    del trainer
    del processor


def train(model_name: str, output_model_name: str, output_dir: str, dataset: Dataset, completion_column: str):
    """
    Qwen2_vl model training pipeline

    Args:
        model_name (str): model to train
        output_model_name (str): model name to output
        output_dir (str): directory to output
        dataset (Dataset): training dataset
        completion_column (str): completion column to use in training dataset
    """
    logger.info(f"▶️ Start qwen2_vl fine tuning pipeline")

    # Load the model and the tokenizer
    model, processor = load_model_and_processor(model_name)

    # Format dataset as conversations in new column
    dataset = construct_conversations(dataset, completion_column=completion_column)

    # Train the model
    trainer = build_trainer(model, processor, dataset, output_dir)
    trainer.train()
    logger.info("✅ Model trained")

    # Save the model
    save_model(trainer, processor, output_model_name, output_dir)

    # Save the instruction
    save_dataset_instruction(dataset, destination=model_get_finetuned_dir(output_model_name))
