docker run -it --name dtop_container_11.8 --gpus all --shm-size=16g \
    -v /tmp2/christine/dtop:/workspace \
    -v /home/christine:/home \
    nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 /bin/bash

apt update && apt install -y vim python3 python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx
pip install uv
pip install openmim setuptools
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
mim install "mmcv==2.0.0"
mim install "mmcls==1.0.0.rc5"
mim install "mmsegmentation==1.0.0rc6"
mim install "mmengine==0.7.0"
pip install einops tomli platformdirs importlib_resources fvcore
pip install timm==1.0.10
pip install yapf==0.40.0