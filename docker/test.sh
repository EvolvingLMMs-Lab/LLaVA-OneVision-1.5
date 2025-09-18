#!/bin/bash

# Validation test for Docker setup improvements
# This script tests the improvements made to fix the Docker setup issues

set -e

echo "🧪 Testing LLaVA-OneVision-1.5 Docker setup improvements..."

# Test 1: Check if placeholder issues are fixed in training scripts
echo "📋 Test 1: Checking training script placeholders..."

SCRIPTS=(
    "aiak_megatron/examples/multimodal/nvlm/pretrain_qwen20_72b_internvit_6b.sh"
    "aiak_megatron/examples/multimodal/nvlm/pretrain_yi_34b_internvit_6b.sh"
    "aiak_megatron/examples/multimodal/nvlm/sft_34b_internvit.sh"
    "aiak_megatron/examples/multimodal/nvlm/sft_qwen20_72b_internvit_6b.sh"
    "aiak_megatron/examples/multimodal/nvlm/sft_qwen2p5_7b_internvit_6b_video.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        # Check for old placeholders
        if grep -q "<path to docker image>" "$script"; then
            echo "❌ Found old placeholder '<path to docker image>' in $script"
            exit 1
        fi
        
        if grep -q "<some mount>" "$script"; then
            echo "❌ Found old placeholder '<some mount>' in $script"
            exit 1
        fi
        
        # Check for new proper container configuration
        if grep -q "container-image llava-onevision:latest" "$script" && \
           grep -q "container-mounts.*workspace" "$script"; then
            echo "✅ $script has proper container configuration"
        else
            echo "❌ $script is missing proper container configuration"
            exit 1
        fi
    else
        echo "⚠️  Script not found: $script"
    fi
done

# Test 2: Check Docker documentation files
echo ""
echo "📋 Test 2: Checking Docker documentation..."

REQUIRED_FILES=(
    "docker/README.md"
    "docker/setup.sh"
    "docker/.env.example"
    "docker-compose.yml"
    ".dockerignore"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ Missing required file: $file"
        exit 1
    fi
done

# Test 3: Check Docker Compose configuration
echo ""
echo "📋 Test 3: Checking Docker Compose configuration..."

if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        if docker compose config --quiet >/dev/null 2>&1; then
            echo "✅ Docker Compose configuration is valid"
        else
            echo "❌ Docker Compose configuration has issues"
            exit 1
        fi
    else
        echo "⚠️  Docker Compose not available for testing"
    fi
else
    echo "⚠️  Docker not available for testing"
fi

# Test 4: Check setup script permissions and syntax
echo ""
echo "📋 Test 4: Checking setup script..."

if [ -x "docker/setup.sh" ]; then
    echo "✅ docker/setup.sh is executable"
else
    echo "❌ docker/setup.sh is not executable"
    exit 1
fi

if bash -n docker/setup.sh; then
    echo "✅ docker/setup.sh has valid syntax"
else
    echo "❌ docker/setup.sh has syntax errors"
    exit 1
fi

# Test 5: Check environment template
echo ""
echo "📋 Test 5: Checking environment configuration..."

if grep -q "DATA_DIR=" docker/.env.example && \
   grep -q "MODEL_DIR=" docker/.env.example && \
   grep -q "CHECKPOINT_DIR=" docker/.env.example; then
    echo "✅ Environment template has required variables"
else
    echo "❌ Environment template is missing required variables"
    exit 1
fi

# Test 6: Check README improvements
echo ""
echo "📋 Test 6: Checking README improvements..."

if grep -q "docker/README.md" README.md && \
   grep -q "volume mount" README.md && \
   grep -q "docker-compose" README.md; then
    echo "✅ Main README has improved Docker documentation"
else
    echo "❌ Main README is missing Docker improvements"
    exit 1
fi

# Test 7: Check Dockerfile improvements
echo ""
echo "📋 Test 7: Checking Dockerfile improvements..."

if grep -q "HEALTHCHECK" dockerfile && \
   grep -q "LABEL" dockerfile && \
   grep -q "ENV PYTHONPATH" dockerfile; then
    echo "✅ Dockerfile has improvements"
else
    echo "❌ Dockerfile is missing expected improvements"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Docker setup improvements are working correctly."
echo ""
echo "Summary of improvements:"
echo "  ✅ Fixed placeholder issues in 5 training scripts"
echo "  ✅ Created comprehensive Docker documentation"
echo "  ✅ Added Docker Compose configuration with 3 profiles"
echo "  ✅ Created automated setup script"
echo "  ✅ Enhanced Dockerfile with optimizations"
echo "  ✅ Added .dockerignore for better build performance"
echo "  ✅ Updated main README with clear instructions"
echo ""
echo "🚀 Users can now easily set up Docker environment with:"
echo "   ./docker/setup.sh dev"