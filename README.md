> build docker image

```make docker-build-finetuning``` --> build the finetuning image
```make docker-build-inference``` --> build the inference image
```make docker-build-inference-app``` --> build the inference app image


```make docker-push-finetuning``` --> push the finetuning image
```make docker-push-inference``` --> push the inference image
```make docker-push-inference-app``` --> push the inference app image



> launch finetuning job on ovh
`ovhai job run  --gpu 1 --volume llm-datasets@1azgra:/workspace/datasets:ro --volume llm-jobs@1azgra:/workspace/jobs:rwd --env HF_TOKEN=<huggingface_token> <other_envs> ghcr.io/dataesr/llm-finetuning:latest -- uv run main.py  --mode ["train", "push"] --model_name <huggingface_model_name> --pipeline ["causallm", "custom"] --dataset_name <dataset_name_from_object_storage> --dataset_config <dataset_config_name> --dataset_format <dataset_format> --push_model_dir <dir_of_finetuned_model>`

examples:
- Simple Fine-Tuning
`ovhai job run --gpu 1 --volume llm-datasets@1azgra:/workspace/datasets:ro --volume llm-jobs@1azgra:/workspace/jobs:rwd --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- uv run main.py --model_name meta-llama/Llama-3.2-1B --dataset_name test.json`

- Fine-tuning and push to hub
`ovhai job run --gpu 1 --volume llm-datasets@1azgra:/workspace/datasets:ro --volume llm-jobs@1azgra:/workspace/jobs:rwd --env HF_TOKEN=hf_abcdefg HF_PUSH_REPO=nanani/nanana ghcr.io/dataesr/llm-finetuning:latest -- uv run main.py --model_name meta-llama/Llama-3.2-1B --dataset_name test.json`

- Push trained model to hub
`ovhai job run --cpu 1 --volume llm-jobs@1azgra:/workspace/jobs:ro --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest HF_PUSH_REPO=nanani/nanana -- uv run main.py --mode push --push_model_dir llama-3.2-1b/finetuned`

- Delete all jobs files
`ovhai bucket object delete llm-jobs@1azgra --all --yes`

> Launch inference job on ovh
`ovhai job run  --gpu 1 --volume llm-datasets@1azgra:/workspace/datasets:ro --volume llm-completions@1azgra:/workspace/completions:rwd --env HF_TOKEN=<huggingface_token> <other_envs> ghcr.io/dataesr/llm-inference:latest -- uv run main.py --model_name <huggingface_model_name> --dataset_name <dataset_name_from_object_storage> --dataset_split <dataset_split> --dataset_config <dataset_config_name>`

> Launch inference app on ovh
`ovhai app run --gpu 1 --env HF_TOKEN=<huggingface_token> --env MODEL_NAME=<huggingface_model_name> --default-http-port 8000 --unsecure-http ghcr.io/dataesr/llm-inference-app:latest`


> Environment variables
General
- HF_TOKEN            = HuggingFace token
- HF_PUSH_REPO        = HuggingFace repository (will try to push finetuned model to repo if defined)
- MLFLOW_TRACKING_URI = Mlflow tracking url (mlflow tracking disabled if not set)
- MLFLOW_RUN_NAME     = Mlflow tracking run name
- MLFLOW_RUN_NAME_TAG = Suffix for auto generated run name if MLFLOW_RUN_NAME not set
- MLFLOW_EXPERIMENT_NAME = Mlflow tracking experiment name
- MLFLOW_MODEL_NAME = Name for model logged into Mlflow

Training params (finetuning)
- MAX_SEQ_LENGTH, NUM_TRAIN_EPOCHS, MAX_STEPS, BATCH_SIZE, GRAD_ACC_STEPS, OPTIM, LEARNING_RATE, LR_SCHEDULER, WEIGHT_DECAY, MAX_GRAD_NORM, WARMUP_RATIO, SAVE_STEPS, LOG_STEPS
- LORA_R, LORA_ALPHA, LORA_DROPOUT, TASK_TYPE
- BNB_4BIT, BNB_QUANT_TYPE, BNB_DOUBLE_QUANT, BNB_COMPUTE_DTYPE