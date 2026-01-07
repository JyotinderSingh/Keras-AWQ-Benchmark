# =============================================================================
# AWQ Benchmarking Docker Image
# =============================================================================
# Pre-built image: jyotindersingh/awq-benchmark
#
# Custom keras/keras-hub can be installed at RUNTIME via environment variables:
#   -e KERAS_SOURCE="git+https://github.com/user/keras@branch"
#   -e KERAS_HUB_SOURCE="git+https://github.com/user/keras-hub@branch"
# =============================================================================

# =============================================================================
# AWQ Benchmarking Docker Image
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG PYTHON_VERSION="3.12"

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV KERAS_BACKEND=tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=2
# FIX FOR PEP 668: Allows pip to install globally in the container
ENV PIP_BREAK_SYSTEM_PACKAGES=1 

# Install system dependencies
# Note: 'python3.12-distutils' removed as it is deprecated/removed in 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
# The --break-system-packages flag is implied by the ENV var set above,
# but can be added explicitly if preferred.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && python -m pip install --upgrade pip setuptools wheel

# Install ML dependencies
RUN pip install --no-cache-dir \
    "tensorflow[and-cuda]>=2.16" \
    keras \
    keras-hub \
    numpy \
    datasets \
    huggingface_hub \
    requests \
    tqdm \
    psutil \
    nvidia-ml-py

# Create workspace
WORKDIR /workspace

# Copy benchmark scripts
COPY scripts/ /workspace/scripts/

# Copy entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create output directory
RUN mkdir -p /workspace/outputs

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]