docker run -it --name dtop_container_11.8 --gpus all --shm-size=16g \
    -v /tmp2/christine/dtop:/workspace \
    -v /home/christine:/home \
    nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 /bin/bash

apt update && apt install -y vim python3 python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx
pip install uv
