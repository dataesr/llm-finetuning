# syntax=docker/dockerfile:1
# FROM nvidia/cuda:12.6.3-base-ubuntu24.04
FROM python:3.11-slim

# # Install necessary packages and CUDA toolkit
# RUN apt-get update && apt-get install -y --no-install-recommends \
#   build-essential \
#   wget \
#   curl \
#   gnupg2 \
#   ca-certificates \
#   && rm -rf /var/lib/apt/lists/*

# # Add CUDA repository key
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# # Add CUDA repository to apt sources list
# RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
#   > /etc/apt/sources.list.d/cuda.list

# # Install CUDA Toolkit 12.6
# RUN apt-get update && apt-get install -y --no-install-recommends \
#   cuda-toolkit-12-6 \
#   && rm -rf /var/lib/apt/lists/*

# # Set the CUDA environment variables
# ENV PATH /usr/local/cuda-12.6/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}

# # Verify the CUDA installation
# RUN apt-get update && apt-get install git-lfs -y && \
#   apt-get install -y libsndfile1 ffmpeg &&  \
#   apt-get clean && rm -rf /var/lib/apt/lists/* &&  \
#   git lfs install

# RUN nvcc --version

WORKDIR /

COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools && pip3 install -r requirements.txt --proxy=${HTTP_PROXY}

# ENV HF_DATASETS_CACHE="/cache/datasets"
# ENV HUGGINGFACE_HUB_CACHE="/cache/hub"
# ENV TRANSFORMERS_CACHE="/cache/transformers"
# ENV TORCH_CACHE="/cache/torch"

RUN mkdir /app
WORKDIR /app

COPY . /app

# ENV PYTHONUNBUFFERED=true

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]