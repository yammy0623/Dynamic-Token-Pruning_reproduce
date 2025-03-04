# Use NVIDIA base image with CUDA 11.8 and Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set the working directory to /workspace
WORKDIR /workspace

# Update package index and install dependencies
RUN apt update && apt install -y \
    vim \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx

# Install Python packages
RUN pip install uv \
    openmim setuptools \
    torch==2.0.0 \
    torchvision==0.15.1 \
    torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
    einops tomli platformdirs importlib_resources fvcore \
    timm==1.0.10 \
    yapf==0.40.0

# Install MMCV, MMClassification, MMSegmentation, and MMEngine using mim
RUN mim install "mmcv==2.0.0" \
    && mim install "mmcls==1.0.0.rc5" \
    && mim install "mmsegmentation==1.0.0rc6" \
    && mim install "mmengine==0.7.0"

# Set up the volume mounts for workspace and user home directories
VOLUME ["/tmp2/christine/dtop", "/home/christine"]

# The container will start by running bash
CMD ["/bin/bash"]
