# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  unzip \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Create and set the HOME directory
WORKDIR /workspace
ENV HOME=/workspace
# Give the OVHcloud user (42420:42420) access to this directory
RUN chown -R 42420:42420 /workspace

# Create python virtual environment
RUN python3 -m venv venv
ENV PATH="./venv/bin:$PATH"

# Install PyTorch for cuda 12.6
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --trusted-host download.pytorch.org
# Install python packages
COPY requirements.txt /workspace
RUN pip install --upgrade pip && pip install -r requirements.txt --proxy=${HTTP_PROXY}

# Add all files to workpspace
ADD . /workspace

CMD ["python3", "script/main.py"]