#!/bin/bash

# Quick setup script for LLaVA-OneVision-1.5 Docker environment
# Usage: ./docker/setup.sh [dev|training|inference]

set -e

# Default values
PROFILE="dev"
BUILD_IMAGE=true
SETUP_ENV=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        dev|training|inference)
            PROFILE="$1"
            shift
            ;;
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        --no-env)
            SETUP_ENV=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [dev|training|inference] [--no-build] [--no-env]"
            echo ""
            echo "Profiles:"
            echo "  dev        - Development environment with source code mounted"
            echo "  training   - Training environment with read-only source"
            echo "  inference  - Inference environment with minimal resources"
            echo ""
            echo "Options:"
            echo "  --no-build - Skip Docker image building"
            echo "  --no-env   - Skip environment file setup"
            echo "  -h, --help - Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Setting up LLaVA-OneVision-1.5 Docker environment..."
echo "Profile: $PROFILE"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Determine Docker Compose command
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check for NVIDIA Docker runtime (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU acceleration will not be available."
    echo "   To install NVIDIA Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Setup environment file
if [ "$SETUP_ENV" = true ]; then
    if [ ! -f .env ]; then
        echo "📝 Creating .env file from template..."
        cp docker/.env.example .env
        echo "✏️  Please edit .env file to configure your data paths:"
        echo "   - DATA_DIR: Path to your training datasets"
        echo "   - MODEL_DIR: Path to your pre-trained models"
        echo "   - CHECKPOINT_DIR: Path for saving checkpoints"
        echo "   - OUTPUT_DIR: Path for output files"
        echo ""
        echo "   Example paths are provided in the .env file."
        read -p "Press Enter to continue after editing .env file..."
    else
        echo "✅ .env file already exists"
    fi
fi

# Build Docker image
if [ "$BUILD_IMAGE" = true ]; then
    echo "🏗️  Building Docker image..."
    docker build -t llava-onevision:latest .
    echo "✅ Docker image built successfully"
else
    echo "⏭️  Skipping Docker image build"
fi

# Setup directories if they don't exist
if [ -f .env ]; then
    echo "📁 Setting up directories..."
    
    # Source environment variables
    set -a
    source .env
    set +a
    
    # Create directories if they don't exist
    for dir in "$DATA_DIR" "$MODEL_DIR" "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$CONFIG_DIR" "$LOG_DIR"; do
        if [ -n "$dir" ] && [ "$dir" != "/path/to/your/"* ]; then
            mkdir -p "$dir" 2>/dev/null || true
        fi
    done
fi

# Start the environment
echo "🐳 Starting Docker environment with profile: $PROFILE"

case $PROFILE in
    dev)
        echo "🔧 Starting development environment..."
        $DOCKER_COMPOSE --profile dev up -d
        echo ""
        echo "✅ Development environment is ready!"
        echo "📋 To access the container:"
        echo "   $DOCKER_COMPOSE exec llava-dev bash"
        echo ""
        echo "🛠️  Available commands inside container:"
        echo "   cd /workspace/LLaVA-OneVision-1.5"
        echo "   python examples/inference/simple_inference.py"
        echo "   bash examples/llava_ov_1_5/pretrain_stage1.sh"
        ;;
    training)
        echo "🏋️  Starting training environment..."
        $DOCKER_COMPOSE --profile training up -d
        echo ""
        echo "✅ Training environment is ready!"
        echo "📋 To access the container:"
        echo "   $DOCKER_COMPOSE exec llava-training bash"
        echo ""
        echo "🚄 To start training:"
        echo "   bash examples/llava_ov_1_5/pretrain_stage1.sh"
        ;;
    inference)
        echo "🔮 Starting inference environment..."
        $DOCKER_COMPOSE --profile inference up -d
        echo ""
        echo "✅ Inference environment is ready!"
        echo "📋 To access the container:"
        echo "   $DOCKER_COMPOSE exec llava-inference bash"
        echo ""
        echo "🌐 Inference server will be available at:"
        echo "   http://localhost:8000"
        ;;
esac

echo ""
echo "📚 For more information, see docker/README.md"
echo "🛑 To stop the environment: $DOCKER_COMPOSE --profile $PROFILE down"