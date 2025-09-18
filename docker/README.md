# Docker Setup Guide for LLaVA-OneVision-1.5

This guide provides comprehensive instructions for setting up and running LLaVA-OneVision-1.5 in Docker containers, with best practices for production environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Images](#docker-images)
- [Volume Mounting Setup](#volume-mounting-setup)
- [Development Environment](#development-environment)
- [Production Deployment](#production-deployment)
- [Training with Docker](#training-with-docker)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- NVIDIA Docker runtime (for GPU support)
- At least 16GB RAM recommended
- NVIDIA GPU with CUDA Compute Capability 7.0+ (for GPU acceleration)

### 2. Build the Docker Image

```bash
# Clone the repository
git clone https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5.git
cd LLaVA-OneVision-1.5

# Build the Docker image
docker build -t llava-onevision:latest .

# Alternative: Build with custom PyPI mirror for faster installation
docker build \
    --build-arg PYPI_MIRROR=https://pypi.org/simple \
    -t llava-onevision:latest .
```

### 3. Run the Container

```bash
# Basic container run (development)
docker run -it --rm \
    --gpus all \
    --ipc host \
    --net host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name llava-container \
    llava-onevision:latest

# Production container run with volume mounts
docker run -d \
    --gpus all \
    --ipc host \
    --net host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name llava-container \
    -v /path/to/your/data:/workspace/data \
    -v /path/to/your/models:/workspace/models \
    -v /path/to/your/checkpoints:/workspace/checkpoints \
    llava-onevision:latest
```

## Docker Images

### Base Image
The default Dockerfile is based on `nvcr.io/nvidia/pytorch:25.04-py3` which includes:
- PyTorch with CUDA support
- Python 3.x
- Essential ML libraries
- CUDA runtime

### Custom Images

#### For CPU-only environments:
```dockerfile
FROM python:3.10-slim
# Add CPU-specific optimizations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
```

#### For specific CUDA versions:
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.04-py3
# Use earlier CUDA version if needed
```

## Volume Mounting Setup

**⚠️ Important:** Use volume mounts instead of symbolic links for better container isolation and portability.

### Recommended Directory Structure

```
/host/path/
├── data/                   # Training datasets
├── models/                 # Pre-trained models
├── checkpoints/           # Training checkpoints
├── outputs/               # Generated outputs
└── configs/               # Configuration files
```

### Volume Mount Examples

```bash
# Complete setup with all volumes
docker run -it --rm \
    --gpus all \
    --ipc host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name llava-training \
    -v /host/data:/workspace/data:ro \
    -v /host/models:/workspace/models:ro \
    -v /host/checkpoints:/workspace/checkpoints \
    -v /host/outputs:/workspace/outputs \
    -v /host/configs:/workspace/configs:ro \
    llava-onevision:latest

# For read-only data and models, use :ro flag for security
# For checkpoints and outputs, use read-write access
```

### Environment Variables

Set these environment variables for consistent paths:

```bash
export LLAVA_DATA_DIR=/workspace/data
export LLAVA_MODEL_DIR=/workspace/models
export LLAVA_CHECKPOINT_DIR=/workspace/checkpoints
export LLAVA_OUTPUT_DIR=/workspace/outputs
```

## Development Environment

### Interactive Development Setup

```bash
# Development container with source code mounted
docker run -it --rm \
    --gpus all \
    --ipc host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name llava-dev \
    -v $(pwd):/workspace/LLaVA-OneVision-1.5 \
    -v /host/data:/workspace/data \
    -v /host/models:/workspace/models \
    -v /host/checkpoints:/workspace/checkpoints \
    -w /workspace/LLaVA-OneVision-1.5 \
    llava-onevision:latest bash
```

### Development with Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  llava-dev:
    build: .
    image: llava-onevision:latest
    container_name: llava-dev
    privileged: true
    cap_add:
      - IPC_LOCK
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    network_mode: host
    volumes:
      - .:/workspace/LLaVA-OneVision-1.5
      - ${DATA_DIR:-./data}:/workspace/data
      - ${MODEL_DIR:-./models}:/workspace/models
      - ${CHECKPOINT_DIR:-./checkpoints}:/workspace/checkpoints
      - ${OUTPUT_DIR:-./outputs}:/workspace/outputs
    working_dir: /workspace/LLaVA-OneVision-1.5
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
      - LLAVA_DATA_DIR=/workspace/data
      - LLAVA_MODEL_DIR=/workspace/models
      - LLAVA_CHECKPOINT_DIR=/workspace/checkpoints
      - LLAVA_OUTPUT_DIR=/workspace/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    stdin_open: true
    tty: true
```

Run with Docker Compose:

```bash
# Set environment variables
export DATA_DIR=/host/path/to/data
export MODEL_DIR=/host/path/to/models
export CHECKPOINT_DIR=/host/path/to/checkpoints
export OUTPUT_DIR=/host/path/to/outputs

# Start development environment
docker-compose up -d llava-dev

# Attach to running container
docker-compose exec llava-dev bash
```

## Production Deployment

### SLURM Integration

For HPC environments with SLURM, update your training scripts with proper container paths:

```bash
#!/bin/bash
#SBATCH --job-name=llava-training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

# Container configuration
CONTAINER_IMAGE="llava-onevision:latest"
CONTAINER_MOUNTS="/host/data:/workspace/data,/host/models:/workspace/models,/host/checkpoints:/workspace/checkpoints"

# Training command
run_cmd="python -u /workspace/LLaVA-OneVision-1.5/examples/multimodal/train.py ${OPTIONS}"

srun -l --verbose \
    --container-image ${CONTAINER_IMAGE} \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --output=${LOGS_DIR}/%x_%j_${DATETIME}.log \
    sh -c "${run_cmd}"
```

### Multi-Node Setup

For distributed training across multiple nodes:

```bash
# On each node, ensure consistent volume mounts
SHARED_DATA="/shared/nfs/data"
SHARED_MODELS="/shared/nfs/models"
SHARED_CHECKPOINTS="/shared/nfs/checkpoints"

docker run -d \
    --gpus all \
    --ipc host \
    --net host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name llava-node-${NODE_ID} \
    -v ${SHARED_DATA}:/workspace/data:ro \
    -v ${SHARED_MODELS}:/workspace/models:ro \
    -v ${SHARED_CHECKPOINTS}:/workspace/checkpoints \
    llava-onevision:latest
```

## Training with Docker

### Stage 1: Alignment Training

```bash
# Set up environment variables
export AIAK_TRAINING_PATH="/workspace/LLaVA-OneVision-1.5"
export DATA_PATH="/workspace/data"
export TOKENIZER_PATH="/workspace/models/tokenizer"
export CHECKPOINT_PATH="/workspace/models/stage0"
export SAVE_CKPT_PATH="/workspace/checkpoints/stage1"

# Run alignment training in container
docker run --rm \
    --gpus all \
    --ipc host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /host/data:/workspace/data \
    -v /host/models:/workspace/models \
    -v /host/checkpoints:/workspace/checkpoints \
    -e AIAK_TRAINING_PATH \
    -e DATA_PATH \
    -e TOKENIZER_PATH \
    -e CHECKPOINT_PATH \
    -e SAVE_CKPT_PATH \
    llava-onevision:latest \
    bash /workspace/LLaVA-OneVision-1.5/examples/llava_ov_1_5/pretrain_stage1.sh
```

### Stage 2: Instruction Tuning

```bash
# Set up environment variables for stage 2
export PRETRAINED_CHECKPOINT="/workspace/checkpoints/stage1"
export SAVE_CKPT_PATH="/workspace/checkpoints/stage2"
export SFT_DATA_PATH="/workspace/data/sft"

# Run instruction tuning
docker run --rm \
    --gpus all \
    --ipc host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /host/data:/workspace/data \
    -v /host/models:/workspace/models \
    -v /host/checkpoints:/workspace/checkpoints \
    -e PRETRAINED_CHECKPOINT \
    -e SAVE_CKPT_PATH \
    -e SFT_DATA_PATH \
    llava-onevision:latest \
    bash /workspace/LLaVA-OneVision-1.5/examples/llava_ov_1_5/sft_stage2.sh
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Accessible
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# If fails, install nvidia-docker2:
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 2. Out of Memory Errors
```bash
# Increase shared memory size
docker run --shm-size=32g ...

# Or use host IPC
docker run --ipc host ...
```

#### 3. Permission Issues with Volumes
```bash
# Fix ownership (run as root in container first)
docker exec -u root llava-container chown -R $(id -u):$(id -g) /workspace/checkpoints

# Or start container with user mapping
docker run --user $(id -u):$(id -g) ...
```

#### 4. Network Connectivity Issues
```bash
# Use host networking for simplicity
docker run --net host ...

# Or configure custom network
docker network create llava-net
docker run --network llava-net ...
```

### Performance Optimization

#### 1. Docker Image Size
```dockerfile
# Multi-stage build to reduce image size
FROM nvcr.io/nvidia/pytorch:25.04-py3 as builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM nvcr.io/nvidia/pytorch:25.04-py3
COPY --from=builder /usr/local/lib/python3.*/site-packages /usr/local/lib/python3.*/site-packages
```

#### 2. Storage Performance
```bash
# Use tmpfs for temporary data
docker run --tmpfs /tmp:size=10g ...

# Use local SSDs for checkpoints
-v /fast/local/ssd:/workspace/checkpoints
```

#### 3. Memory Management
```bash
# Set memory limits
docker run --memory=32g --memory-swap=64g ...

# Monitor memory usage
docker stats llava-container
```

### Debugging

#### 1. Container Inspection
```bash
# Check container logs
docker logs llava-container

# Execute commands in running container
docker exec -it llava-container bash

# Inspect container configuration
docker inspect llava-container
```

#### 2. Resource Monitoring
```bash
# Monitor GPU usage
docker exec llava-container nvidia-smi

# Monitor system resources
docker exec llava-container htop
```

For additional support, please refer to the main repository documentation or open an issue on GitHub.