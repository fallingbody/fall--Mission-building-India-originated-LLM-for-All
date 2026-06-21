FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl \
    build-essential cmake \
    libopenmpi-dev \
    nmap metasploit-framework hashcat john hydra \
    smartmontools lm-sensors \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir vllm mamba-ssm transformer-engine

# FALL source
COPY . /app/FALL
RUN pip install -e /app/FALL

# Entry point
ENTRYPOINT ["python", "-m", "fall"]