# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

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

COPY requirements.txt .

# Create python virtual environment
RUN python3 -m venv venv
ENV PATH="./venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt --proxy=${HTTP_PROXY}

RUN mkdir /app
COPY app ./app

EXPOSE 8080

CMD ["./venv/bin/fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8080"]