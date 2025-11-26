# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  unzip \
  git \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Give the OVHcloud user (42420:42420) access  
USER 42420:42420

# Create and set the HOME directory
WORKDIR /workspace
ENV HOME=/workspace
# Give the OVHcloud user (42420:42420) access to this directory
# RUN chown -R 42420:42420 /workspace

# Create python virtual environment
# RUN python3 -m venv venv
# ENV PATH="./venv/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="$HOME/.local/bin/:$PATH"
RUN uv venv --python 3.12 --seed
ENV PATH="$HOME/.venv/bin:$PATH"

# Install PyTorch for cuda 12.8
# RUN pip install --upgrade pip 
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --trusted-host download.pytorch.org
# ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# RUN uv pip install vllm==0.10.2 --torch-backend=cu128

# Install python packages
COPY finetuning/requirements.txt /workspace
RUN uv pip install -r requirements.txt

# Clean cache
RUN uv cache clean

# Add all files to workpspace
COPY shared /workspace/shared
COPY finetuning /workspace

# Clean cache and make sure user as rights
# RUN uv cache clean && chown -R 42420:42420 /workspace/.cache

# CMD ["python3", "train.py"]
# CMD ["uv", "run", "train.py"]