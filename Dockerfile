# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

WORKDIR /

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  unzip \
  python3 \
  python3-pip \
  python3-venv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
  unzip awscliv2.zip && \
  ./aws/install && \
  rm -rf awscliv2.zip aws/

# Disable python buffering
ENV PYTHONUNBUFFERED=1
# Needed for python env (?)
ENV CUDA_VISIBLE_DEVICES=0

# Create python virtual environment
RUN python3 -m venv venv
ENV PATH="./venv/bin:$PATH"

# Install PyTorch for cuda 12.6
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org
# Install python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt --proxy=${HTTP_PROXY}

# Copy application code
RUN mkdir /app
COPY app ./app

# use ./venv/bin/fastapi if nvidia/cuda:12.6.3-base-ubuntu24.04
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0"]