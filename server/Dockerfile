# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

WORKDIR /app

# Install python
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  python3-venv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Disable python buffering
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# Create python virtual environment
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt --proxy=${HTTP_PROXY}

ENV FLASK_DEBUG=1

COPY . .

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

EXPOSE 5000

ENTRYPOINT ["./venv/bin/python3", "run.py", "run"]