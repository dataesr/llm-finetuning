# llm-finetuning docker image

```make docker-build``` --> build the image

```make docker-push``` --> push the image

# llm-finetuning script

`ovhai job run  --gpu 1 --volume llm-datasets@GRA:/workspace/datasets:ro --volume llm-jobs@GRA:/workspace/jobs:rw --env HF_TOKEN=<huggingface_token> ghcr.io/dataesr/llm-finetuning:latest -- ./venv/bin/python3 script/main.py  --mode ["train", "push", "delete"] --model_name <huggingface_model_name> --dataset_name <dataset_name_from_object_storage> --output_model_name <name_of_finetuned_model> --hf_hub <huggingface_hub_id> --hf_hub_private`

examples:
- Simple Fine-Tuning
`ovhai job run --gpu 1 --volume llm-datasets@GRA:/workspace/datasets:ro --volume llm-jobs@GRA:/workspace/jobs:rw --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- ./venv/bin/python3 script/main.py --model_name meta-llama/Llama-3.2-1B --dataset_name test.json --output_model_name Llama-3.2-1B-finetuned`

- Fine-tuning and push to hub
`ovhai job run --gpu 1 --volume llm-datasets@GRA:/workspace/datasets:ro --volume llm-jobs@GRA:/workspace/jobs:rw --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- ./venv/bin/python3 script/main.py --model_name meta-llama/Llama-3.2-1B --dataset_name test.json --output_model_name Llama-3.2-1B-finetuned --hf_hub dataesr/hub_name --hf_hub_private`

- Push existing model to hub
`ovhai job run --cpu 1 --volume llm-jobs@GRA:/workspace/jobs:ro --env HF_TOKEN=hf_abcdefg ghcr.io/dataesr/llm-finetuning:latest -- ./venv/bin/python3 script/main.py --mode push --output_model_name Llama-3.2-1B-finetuned --hf_hub dataesr/hub_name --hf_hub_private`

- Delete model files
`ovhai job run --cpu 1 --volume llm-jobs@GRA:/workspace/jobs:rw ghcr.io/dataesr/llm-finetuning:latest -- ./venv/bin/python3 script/main.py --mode delete --output_model_name Llama-3.2-1B-finetuned`