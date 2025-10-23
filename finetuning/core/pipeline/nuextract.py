import torch
from transformers import (
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from datasets import Dataset
from core.utils import model_get_finetuned_dir, model_get_extracted_dir
from shared.dataset import (
    INSTRUCTION_FIELD,
    INPUT_FIELD,
    COMPLETION_FIELD,
    CHAT_TEMPLATE_FIELD,
    CONVERSATIONS_FIELD,
)
from shared.logger import get_logger

logger = get_logger(__name__)

# Training arguments (https://huggingface.co/docs/transformers/en/main_classes/trainer)
# https://github.com/numindai/nuextract/blob/main/cookbooks/nuextract-2.0_sft.ipynb
num_train_epochs = 1  # Number of training epochs
max_steps = 300  # Number of training steps, should be set to -1 for full training
per_device_train_batch_size = 1  # Batch size per device during training. Optimal given our GPU vram.
gradient_accumulation_steps = 4  # Number of steps before performing a backward/update pass
optim = "paged_adamw_8bit"
learning_rate = 2e-5  # The initial learning rate for AdamW optimizer
lr_scheduler_type = "linear"  # Scheduler rate type
max_seq_length = 8192  # Context window length.

# BitsAndBytesConfig int-4 config (https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/quantization#transformers.BitsAndBytesConfig)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
)

# LORA config (https://huggingface.co/docs/peft/package_reference/lora)
lora_config = LoraConfig(
    r=8,
    lora_alpha=64,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # for full training
)

default_chat_template = "{%- for message in messages -%}\n    {#--- Handle User Messages with Template ---#}\n    {%- if message['role'] == 'user' -%}\n        {%- if loop.first -%}\n            {{- '<|im_start|>system\n' }}\n            {{- 'You are NuExtract, an information extraction tool created by NuMind.\n' }}\n            {{- '<|im_end|>\n' }}\n        {%- endif -%}\n        {{- '<|im_start|>' + message['role'] }}\n        {{- '\n# Template:\n' }}\n     {{%- raw -%}}{\"entities\": [{\"entity\": \"verbatim-string\", \"entity_type\": [\"RESEARCH_INFRASTRUCTURE\", \"FUNDER\", \"PRIVATE_COMPANY\"], \"grant_ids\": [\"verbatim-string\"], \"grant_programs\": [\"verbatim-string\"], \"other_ids\": [\"verbatim-string\"]}]}{{%- endraw -%}} \n        {{- '\n# Input:\n' }}\n        {{- message['content'] | trim }}\n        {{- '\n<|im_end|>\n' }}\n    {#--- Handle All Other Messages (Assistant, System, etc.) ---#}\n    {%- else -%}\n        {{- '<|im_start|>' + message['role'] + '\n' }}\n        {{- message['content'] | trim }}\n        {%- if loop.last and message['role'] == 'assistant' -%}\n            {{- '\n<|endoftext|>' }}\n        {%- else -%}\n            {{- '\n<|im_end|>\n' }}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{#--- Add Generation Prompt if Requested ---#}\n{%- if add_generation_prompt -%}\n    {{- '<|im_start|>assistant' }}\n{%- endif -%}"


def extract_text_model_from_vision(
    vision_model_name, output_dir, custom_chat_template=None, base_text_tokenizer: str = "Qwen/Qwen2-7B"
):
    """
    Extract text-only model from vision model and save it

    Args:
    - vision_model_path (str): Path to vision model
    - output_path (str): Path to save text-only model
    - custom_chat_template: Custom chat template
    - base_text_tokenizer (str): model for base tokenizer

    Returns:
    - tokenizer: Tokenizer with updated template
    """
    logger.info(f"Extracting text model from {vision_model_name}")

    # Load vision model and processor (with quantization)
    # processor = AutoProcessor.from_pretrained(
    #     vision_model_name,
    #     trust_remote_code=True,
    #     padding_side="right",
    #     use_fast=True,
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        base_text_tokenizer, trust_remote_code=True, padding_side="right", use_fast=True
    )

    vision_model = AutoModelForVision2Seq.from_pretrained(
        vision_model_name,
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        device_map="auto",
    )

    # Extract the language model (to cpu)
    vision_model = vision_model.cpu()
    language_model = vision_model.language_model.cpu()
    lm_head = vision_model.lm_head.cpu()

    logger.info(f"Extracted language_model: {type(language_model)}")
    logger.info(f"Language model parameters: {language_model.num_parameters():,}")

    # Extract config
    text_config = vision_model.config.text_config

    # Handle rope_scaling - filter out incompatible keys
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling and isinstance(rope_scaling, dict):
        # Remove mrope_section and other VL-specific rope scaling parameters
        rope_scaling = {k: v for k, v in rope_scaling.items() if k not in ["mrope_section"]}
        # If rope_scaling becomes empty or only has unsupported keys, set to None
        if not rope_scaling or rope_scaling.get("rope_type") == "default":
            rope_scaling = None

    # Create a proper Qwen2Config from the text config
    qwen2_config = Qwen2Config(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_hidden_layers=text_config.num_hidden_layers,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=getattr(text_config, "num_key_value_heads", text_config.num_attention_heads),
        hidden_act=text_config.hidden_act,
        max_position_embeddings=text_config.max_position_embeddings,
        initializer_range=text_config.initializer_range,
        rms_norm_eps=text_config.rms_norm_eps,
        use_cache=getattr(text_config, "use_cache", True),
        tie_word_embeddings=getattr(text_config, "tie_word_embeddings", False),
        rope_theta=getattr(text_config, "rope_theta", 10000.0),
        rope_scaling=rope_scaling,
        attention_dropout=getattr(text_config, "attention_dropout", 0.0),
        use_sliding_window=getattr(text_config, "use_sliding_window", False),
        max_window_layers=getattr(text_config, "max_window_layers", 28),
        sliding_window=getattr(text_config, "sliding_window", None),
    )

    # Create a new Qwen2ForCausalLM model with the proper config (on CPU)
    qwen2_model = Qwen2ForCausalLM(qwen2_config).cpu()

    # Combine them with proper prefixes
    qwen2_state_dict = {}
    # Add text_model components with "model." prefix
    for key, value in language_model.state_dict().items():
        # Ensure tensor is on CPU
        qwen2_state_dict[f"model.{key}"] = value.cpu() if hasattr(value, "cpu") else value
    # Add lm_head components with "lm_head." prefix
    for key, value in lm_head.state_dict().items():
        # Ensure tensor is on CPU
        qwen2_state_dict[f"lm_head.{key}"] = value.cpu() if hasattr(value, "cpu") else value

    # Load the fixed state dict
    qwen2_model.load_state_dict(qwen2_state_dict)

    # Set simplified chat template (no image handling)
    if custom_chat_template:
        tokenizer.chat_template = custom_chat_template
        logger.debug(f"Custom chat template: {tokenizer.chat_template}")
        logger.info("✅ Applied custom chat template")

    # Update tokenizer settings
    # tokenizer.eos_token = processor.tokenizer.eos_token
    # tokenizer.eos_token_id = processor.tokenizer.eos_token_id

    # Save the extracted text model and tokenizer
    qwen2_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ Text-only Qwen2 model saved to {output_dir}")

    # Cleanup
    del vision_model
    del language_model
    del lm_head
    del qwen2_config
    del qwen2_model

    return tokenizer


def load_model_and_tokenizer(model_name: str, output_model_name: str, custom_chat_template=None):
    """
    Load model and tokenizer

    Args:
    - model_name (str): Model to load

    Returns:
    - model: Loaded model
    - tokenizer: Loaded tokenizer
    """

    logger.info(f"Start loading model {model_name}")
    extracted_dir = model_get_extracted_dir(output_model_name)
    tokenizer = extract_text_model_from_vision(model_name, extracted_dir, custom_chat_template)

    # Reload with quantization
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        extracted_dir,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )
    logger.info("Reloaded text model with quantization")

    # Configure for training
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.pad_token_id = tokenizer.eos_token_id

    # Prepare model for lora
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    logger.debug(f"Model embeddings size: {model.get_input_embeddings().weight.size(0)}")
    logger.debug(f"Tokenizer chat template: {tokenizer.chat_template}")
    logger.info(f"✅ Model and tokenizer loaded")

    return model, tokenizer


def construct_one_conversation(user: str, system: str = None, assistant: str = None):
    """
    Construct a conversation from system, user and assistant messages

    Args:
    - system (str): system instructions
    - user (str): user input
    - template (str): extraction template (for NuExtract)
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
        conversation.append({"role": "assistant", "content": f"{assistant}"})

    return conversation


def construct_conversations(dataset: Dataset, custom_instruction: str = None) -> Dataset:
    """
    Construct conversations style column for training

    Args:
    - dataset (Dataset): training dataset
    - custom_instruction (str): custom system prompt

    Returns the training dataset with a conversations column
    """

    def map_conversations(example):
        return {
            CONVERSATIONS_FIELD: construct_one_conversation(
                system=custom_instruction,
                user=example[INPUT_FIELD],
                assistant=example[COMPLETION_FIELD],
            )
        }

    dataset = dataset.map(map_conversations).select_columns([CONVERSATIONS_FIELD])
    logger.debug(f"✅ Dataset formatted with conversation format")
    logger.debug(f"Dataset columns: {dataset.column_names}")
    logger.debug(f"Dataset conversations sample: {dataset[0][CONVERSATIONS_FIELD]}")
    return dataset


def build_trainer(model, tokenizer, dataset: Dataset, output_dir: str) -> SFTTrainer:
    """
    Build SFTTrainer for finetuning

    Args:
    - model: text model
    - tokenizer: tokenizer
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
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        optim=optim,
        save_steps=200,
        logging_steps=10,
        logging_strategy="steps",
        logging_first_step=True,
        disable_tqdm=False,
        report_to="wandb",
        dataloader_pin_memory=False,
        # max_seq_length=max_seq_length,
        # packing = False,
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
    - processor: processor to save
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
    NueExtract model training pipeline

    Args:
        model_name (str): model to train
        output_model_name (str): model name to output
        output_dir (str): directory to output
        dataset (Dataset): training dataset
    """
    logger.info(f"▶️ Start NueExtract fine tuning pipeline")

    # Load the model and the tokenizer
    custom_instruction = kwargs.get("dataset_extras", {}).get(INSTRUCTION_FIELD)
    custom_chat_template = kwargs.get("dataset_extras", {}).get(CHAT_TEMPLATE_FIELD)
    model, tokenizer = load_model_and_tokenizer(model_name, output_model_name, custom_chat_template=custom_chat_template)

    # Format dataset as conversations in new column
    dataset = construct_conversations(dataset, custom_instruction=custom_instruction)
    logger.debug(f"Chat template sample: {tokenizer.apply_chat_template(dataset[0][CONVERSATIONS_FIELD], tokenize=False)}")

    # Train the model
    trainer = build_trainer(model, tokenizer, dataset, output_dir)
    trainer.train()
    logger.info("✅ Model trained")

    # Save the model
    merge_and_save_model(trainer, tokenizer, output_model_name, output_dir=output_dir)

    # Save the instruction
    # save_dataset_instruction(dataset, destination=model_get_finetuned_dir(output_model_name))
