# Migration Guide: Docker Setup Improvements

This guide helps users migrate from the old Docker setup to the new improved setup.

## What Changed

### 1. Training Scripts
**Before:** Scripts contained placeholders like `<path to docker image>` and `<some mount>`
**After:** Scripts use proper image names and environment-based volume mounts

### 2. Docker Setup
**Before:** Basic docker run commands with limited documentation
**After:** Comprehensive setup with Docker Compose, environment templates, and automated scripts

### 3. Volume Management
**Before:** Unclear guidance on data mounting
**After:** Clear volume mounting strategy with dedicated directories

## Migration Steps

### For Existing Users

1. **Update your training scripts** (if you customized them):
   ```bash
   # Old format (problematic):
   --container-image <path to docker image>
   --container-mounts "<some mount>"
   
   # New format (working):
   --container-image llava-onevision:latest
   --container-mounts "${DATA_PATH}:/workspace/data,${MODEL_PATH}:/workspace/models,${CHECKPOINT_DIR}:/workspace/checkpoints,${LOGS_DIR}:/workspace/logs"
   ```

2. **Set up environment variables**:
   ```bash
   # Copy template and customize
   cp docker/.env.example .env
   # Edit .env with your actual paths
   ```

3. **Use new setup methods**:
   ```bash
   # Option 1: Automated setup
   ./docker/setup.sh dev
   
   # Option 2: Docker Compose
   docker-compose --profile dev up -d
   
   # Option 3: Manual (compatible with old method)
   docker build -t llava-onevision:latest .
   docker run -it --rm --gpus all \
     --ipc host --net host --privileged --cap-add IPC_LOCK \
     --ulimit memlock=-1 --ulimit stack=67108864 \
     -v /your/data:/workspace/data \
     -v /your/models:/workspace/models \
     -v /your/checkpoints:/workspace/checkpoints \
     llava-onevision:latest
   ```

### For New Users

Simply follow the new documentation:
1. Read `docker/README.md` for comprehensive guide
2. Use `./docker/setup.sh dev` for quick start
3. Customize `.env` file for your environment

## Benefits of New Setup

1. **No More Placeholders**: Training scripts work out of the box
2. **Better Isolation**: Volume mounts instead of symbolic links
3. **Environment Profiles**: Different configurations for dev/training/inference
4. **Automated Setup**: One command to get started
5. **Better Documentation**: Comprehensive guides and troubleshooting
6. **Production Ready**: Proper security and resource management

## Compatibility

- ✅ Existing Docker commands still work
- ✅ Environment variables are backward compatible
- ✅ No breaking changes to core functionality
- ✅ Old Dockerfile approach still supported

## Getting Help

- Check `docker/README.md` for detailed documentation
- Run `./docker/setup.sh --help` for setup options
- Run `./docker/test.sh` to validate your setup
- Open an issue if you encounter problems

## Quick Reference

```bash
# Development environment
./docker/setup.sh dev

# Training environment  
./docker/setup.sh training

# Inference environment
./docker/setup.sh inference

# Manual Docker Compose
docker-compose --profile dev up -d
docker-compose exec llava-dev bash

# Stop environment
docker-compose --profile dev down
```