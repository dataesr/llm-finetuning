import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import SFTConfig
from project.pipeline.trainer import initialize, train_model, save_model
from project.dataset import get_dataset, save_dataset_instruction, TEXT_FIELD
from project.logger import get_logger


logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
# https://github.com/numindai/nuextract/blob/main/cookbooks/nuextract-2.0_sft.ipynb
max_steps = 600  # Was originally trained on 3000 but better to keep the value low for test purposes.
# Context window length. Llama can now technically push further, but I find attention is no longer working so well.
max_seq_length = 8192

num_train_epochs = 5  # Number of training epochs
per_device_train_batch_size = 1  # Batch size per device during training. Optimal given our GPU vram.
gradient_accumulation_steps = 4  # Number of steps before performing a backward/update pass
gradient_checkpointing = True  # Use gradient checkpointing to save memory
gradient_checkpointing_kwargs = ({"use_reentrant": False},)  # Options for gradient checkpointing
learning_rate = 1e-5  # The initial learning rate for AdamW optimizer
lr_scheduler_type = "constant"  # Scheduler rate type
fp16 = False  # Dno not use fp16 precision
bf16 = True  # Use bfloat16 precision
max_grad_norm = 0.3  # Max gradient norm
warmup_ratio = 0.03  # Warmup ratio
report_to = "none"  # Report metrics to tensorboard
logging_steps = 10  # Log every 10 step
save_steps = 200  # We're going to save the model weights every 200 steps to save our checkpoint


def load_pretrained_model(model_name: str):
    """
    Load pretrained multimodal model from huggingface

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
        attn_implementation="flash_attention_2",
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

    torch.cuda.empty_cache()

    logger.info(f"✅ Model and tokenizer loaded")

    return model, processor


def build_collate_fn(processor):
    """
    Get data collator function
    """

    def collate_fn(inputs):
        # process input/prompt part of conversations
        user_texts = [processor.apply_chat_template(input[:1], tokenize=False) for input in inputs]

        # process full conversations (user + assistant)
        full_texts = [processor.apply_chat_template(input, tokenize=False) for input in inputs]

        # process images
        images = None

        # tokenize sequences
        user_batch = processor(text=user_texts, images=images, return_tensors="pt", padding=True)
        full_batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)

        # mask padding tokens
        labels = full_batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # mask user message tokens for each example in the batch
        for i in range(len(inputs)):
            # length of prompt message (accounting for possible padding)
            user_len = user_batch["attention_mask"][i].sum().item()

            # mask prompt part of label
            labels[i, : user_len - 1] = -100

        full_batch["labels"] = labels
        return full_batch

    return collate_fn


def build_sft_config(output_dir: str):
    """
    Get sft config (training args)
    """
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        max_seq_length=max_seq_length,
        dataset_text_field=TEXT_FIELD,
    )
    return sft_config


def fine_tune(model_name: str, dataset_name: str, output_model_name: str = None):
    """
    Fine-tuning pipeline

    Args:
    - model_name (str): Model to fine-tune
    - dataset_name (str): Dataset to use for fine-tuning
    - output_model_name (str): Fine-tuned model name. Default to None
    """

    logger.info(f"Start fine tuning of model {model_name} with dataset {dataset_name}")

    # Initialize llm folder
    output_model_name, output_dir = initialize(model_name, output_model_name)

    # Load the model and the processor
    model, processor = load_pretrained_model(model_name)

    # Load dataset
    dataset = get_dataset(dataset_name, tokenizer=processor.tokenizer, use_chatml=True)  # chatml mandatory

    # Build collate function
    collate_fn = build_collate_fn(processor)

    # Build sft_config
    sft_config = build_sft_config(output_dir)

    # Train the model
    trainer = train_model(
        model,
        tokenizer=processor.tokenizer,
        dataset=dataset,
        sft_config=sft_config,
        data_collator=collate_fn,
    )

    # Save the model
    save_model(trainer, output_dir=output_dir, output_model_name=output_model_name, processor=processor)

    # Save the instruction
    save_dataset_instruction(dataset, output_dir=output_dir)

    return output_model_name
