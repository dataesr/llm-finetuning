> build docker image

```make docker-build-finetuning``` --> build the finetuning image
```make docker-build-inference``` --> build the inference image

```make docker-push-finetuning``` --> push the finetuning image
```make docker-push-inference``` --> push the inference image


> launch finetuning job on ovh

`ovhai job run  --gpu 1 --volume llm-datasets@1azgra:/workspace/datasets:ro --volume llm-jobs@1azgra:/workspace/jobs:rwd --env HF_TOKEN=<huggingface_token> ghcr.io/dataesr/llm-finetuning:latest -- uv run train.py  --mode ["train", "push"] --model_name <huggingface_model_name> --dataset_name <dataset_name_from_object_storage> --output_model_name <name_of_finetuned_model> --hf_hub <huggingface_hub_id> --hf_hub_private`

examples:
- Simple Fine-Tuning
`ovhai job run --gpu 1 --volume llm-datasets@1azgra:/workspace/datasets:ro --volume llm-jobs@1azgra:/workspace/jobs:rwd --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- uv run train.py --model_name meta-llama/Llama-3.2-1B --dataset_name test.json --output_model_name llama-training`

- Fine-tuning and push to hub
`ovhai job run --gpu 1 --volume llm-datasets@1azgra:/workspace/datasets:ro --volume llm-jobs@1azgra:/workspace/jobs:rwd --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- uv run train.py --model_name meta-llama/Llama-3.2-1B --dataset_name test.json --output_model_name llama-training --hf_hub dataesr/hub_name --hf_hub_private`

- Push trained model to hub
`ovhai job run --cpu 1 --volume llm-jobs@1azgra:/workspace/jobs:ro --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- uv run train.py --mode push --output_model_name llama-training --hf_hub dataesr/hub_name --hf_hub_private`

- Delete all jobs files
`ovhai bucket object delete llm-jobs@1azgra --all --yes`


> Launch inference app on ovh
`ovhai app run --gpu 1 --env HF_TOKEN=<huggingface_token> --env MODEL_NAME=<huggingface_model_name>--default-http-port 8000 --unsecure-http ghcr.io/dataesr/llm-inference:latest`

`ovhai app run --gpu 1 --env HF_TOKEN=hf_abcdef --env MODEL_NAME=dataesr/openchat-3.6-8b-acknowledgments --default-http-port 8000 --unsecure-http ghcr.io/dataesr/llm-inference:latest`