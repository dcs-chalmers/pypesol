FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System packages + solvers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    pkg-config \
    git \
    curl \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    glpk-utils \
    coinor-cbc \
 && rm -rf /var/lib/apt/lists/*

# Copy just requirements first
COPY requirements.txt /app/requirements.txt

# Upgrade pip tooling and install Python deps
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# To be able to run the getting-started tutorial
RUN pip install notebook jupyterlab

# Copy the rest
COPY . /app

# Default command (change as needed)
CMD ["bash"]
