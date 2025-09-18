# LLaVA-OneVision-1.5 Docker Image
# Based on NVIDIA PyTorch container with CUDA support
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Build arguments for customization
ARG PYPI_MIRROR=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG DEBIAN_FRONTEND=noninteractive

# Environment variables
ENV PYTHONPATH="/workspace/LLaVA-OneVision-1.5:$PYTHONPATH"
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# Clear pip constraints if needed (for compatibility with base image)
ARG CLEAR_PIP_CONSTRAINTS=1
RUN set -eux; \
    if [ "${CLEAR_PIP_CONSTRAINTS}" = "1" ] && [ -f /etc/pip/constraint.txt ]; then \
        cp /etc/pip/constraint.txt /etc/pip/constraint.txt.bak; \
        : > /etc/pip/constraint.txt; \
    fi

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        jq \
        vim \
        less \
        rsync \
        wget \
        curl \
        ca-certificates \
        openssh-client \
        htop \
        tree \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -i ${PYPI_MIRROR} \
    ${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
    -r /workspace/requirements.txt

# Create directory structure for volume mounts
RUN mkdir -p /workspace/data \
    /workspace/models \
    /workspace/checkpoints \
    /workspace/outputs \
    /workspace/configs \
    /workspace/logs

# Clean up installation files and unnecessary packages
RUN rm -rf /workspace/requirements.txt \
    /workspace/README.md \
    /workspace/docker-examples \
    /workspace/license.txt \
    /workspace/tutorials \
    /tmp/* \
    /var/tmp/* \
    ~/.cache/pip/*

# Create non-root user for security (optional, can be overridden)
RUN groupadd -r llava && useradd -r -g llava llava

# Set permissions for workspace
RUN chown -R llava:llava /workspace

# Health check to ensure the container is working properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')" || exit 1

# Default command
CMD ["/bin/bash"]

# Add labels for better container management
LABEL maintainer="LLaVA-OneVision Team"
LABEL description="LLaVA-OneVision-1.5 Training and Inference Environment"
LABEL version="1.5"
LABEL gpu.required="true"