# llm-finetuning docker image

```make docker-build``` --> build the image

```make docker-push``` --> push the image

# llm-finetuning script

```ovhai job run <registry-address>/<docker_image_name>:<tag-name> \
      --gpu <nb_gpus> \
      --volume <object_storage_name>@<region>/:/workspace/<mount_directory_name>:<permission_mode> \
      -- bash -c 'pip install -r /workspace/mount_directory_name/requirements.txt && python /workspace/mount_directory_name/python_script.py'```