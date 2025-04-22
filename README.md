# llm-finetuning docker image

```make docker-build``` --> build the image

```make docker-push``` --> push the image

# llm-finetuning script

`ovhai job run  --gpu 1 --volume llm-datasets@GRA:/workspace/datasets:ro --volume llm-jobs@GRA:/workspace/jobs:rw --env HF_TOKEN=<huggingface_token>ghcr.io/dataesr/llm-finetuning:latest -- ./venv/bin/python3 script/main.py --model_name <huggingface_model_name> --dataset_name <dataset_name_from_object_storage> --output_model_name <name_of_finetuned_model> --hf_hub <huggingface_hub_id>`

example:

`ovhai job run --gpu 1 --volume llm-datasets@GRA:/workspace/datasets:ro --volume llm-jobs@GRA:/workspace/jobs:rw --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- ./venv/bin/python3 script/main.py --model_name meta-llama/Llama-3.2-1B --dataset_name test.json --output_model_name Llama-3.2-1B-finetuned`