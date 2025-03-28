# Official Pytorch Implementation of DToP

### Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation 
Quan Tang, Bowen Zhang, Jiajun Liu, Fagui Liu, Yifan Liu

ICCV 2023. [[arxiv]](https://arxiv.org/abs/2308.01045)

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for DToP

As shown in the following figure, the network is naturally split into stages using inherent auxiliary blocks.

<img src="./resources/fig-1-1.png">

## Highlights
* **Dynamic Token Pruning** We introduce a dynamic token pruning paradigm based on the early exit of easy-to-recognize tokens for semantic segmentation transformers.
* **Controllable prune ratio** One hyperparameter to control the trade-off between computation cost and accuracy.
* **Generally applicable** e apply DToP to mainstream semantic segmentation transformers and can reduce up to 35% computational cost without a notable accuracy drop.

## Getting started 
1. requirements
```
torch==2.0.0 mmcls==1.0.0.rc5, mmcv==2.0.0 mmengine==0.7.0 mmsegmentation==1.0.0rc6 
```
or up-to-date mmxx series till 9 Aug 2023

```
ADD
pip install openmim setuptools
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
mim install "mmcv==2.0.0"
mim install "mmcls==1.0.0.rc5"
mim install "mmsegmentation==1.0.0rc6"
mim install "mmengine==0.7.0"
pip install einops
pip install tomli
pip install platformdirs
pip install importlib_resources
pip install fvcore
pip install einops tomli platformdirs importlib_resources fvcore
pip install timm==1.0.10
pip install yapf==0.40.0


run
./tools/dist_train.sh config/prune/BASE_segvit_ade20k.py ./exp
./tools/dist_train_load.sh config/prune/prune_segvit_ade20k.py ./exp ./exp/ckpt-BASE_segvit_ade20k/iter_40000.pth
./tools/dist_test.sh  config/prune/prune_segvit_ade20k_large.py ./exp/ckpt-BASE_segvit_ade20k/iter_40000.pth

data prepare
wget -P ./data http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget -P ./data http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
wget -P ./data http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar
wget -P ./data http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip
wget -P ./data/coco_stuff10k http://images.cocodataset.org/zips/train2017.zip
wget -P ./data/coco_stuff10k http://images.cocodataset.org/zips/val2017.zip

docker build -t dtop_image .
docker run -it --name dtop_container_new --runtime=nvidia --gpus all --shm-size=16g \
    --device=/dev/nvidia-uvm \
    --device=/dev/nvidia-uvm-tools \
    --device=/dev/nvidia-modeset \
    --device=/dev/nvidiactl \
    --device=/dev/nvidia0 \
    --device=/dev/nvidia1 \
    --device=/dev/nvidia2 \
    --device=/dev/nvidia3 \
    -v /tmp2/christine/dtop:/workspace \
    -v /home/christine:/home \
    dtop_image /bin/bash
    
```

## Issue
```
AttributeError: class `PascalContextDataset59` in mmseg/datasets/pascal_context.py: 'PascalContextDataset59' object has no attribute 'file_client'
```
mmseg/datasets/pascal_context.py command 掉 file_client

for COCO dataset
要自己額外去官網加這個轉換code到tool/dataset_converters
https://github.com/open-mmlab/mmsegmentation/blob/0.x/tools/convert_datasets/coco_stuff10k.py

```
# download
mkdir coco_stuff10k && cd coco_stuff10k
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip

# unzip
unzip cocostuff-10k-v1.1.zip

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/coco_stuff10k.py /path/to/coco_stuff10k --nproc 8
python ./tools/dataset_converters/coco_stuff10k.py ./data/coco_stuff10k --nproc 8
```

source: http://mmsegmentation.readthedocs.io/en/0.x/dataset_prepare.html



## Training
To aquire the base model
```
python tools dist_train.sh config/prune/BASE_segvit_ade20k_large.py  $work_dirs$
```
To prune on the base model
```
python tools dist_train_load.sh  config/prune/prune_segvit_ade20k_large.py  $work_dirs$  $path_to_ckpt$
```

## Eval
```
python tools/dist_test.sh  config/prune/prune_segvit_ade20k_large.py  $path_to_ckpt$
```

## Datasets
Please follow the instructions of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) data preparation

## Results
### Ade20k
| Method       | Backbone  | mIoU | GFlops | config | ckpt |
|--------------|-----------|------|--------|--------|------|
| Segvit       | Vit-base  | 49.6 | 109.9  | [config](./config/prune/BASE_segvit_ade20k.py)       |      |
| Segvit-prune | Vit-base | 49.8 |   86.8 | [config](./config/prune/prune_segvit_ade20k.py)       |      |
| Segvit       | Vit-large | 53.3 |  617.0 | [config](./config/prune/BASE_segvit_ade20k_large.py)       |      |
| Segvit-prune | Vit-large | 52.8 |  412.8 |  [config](./config/prune/prune_segvit_ade20k_large.py)      |      |

### Pascal Context
| Method       | Backbone  | mIoU | GFlops | config | ckpt |
|--------------|-----------|------|--------|--------|------|
| Segvit       | Vit-large | 63.0 |  315.4 | [config](./config/prune/BASE_segvit_pc.py)       |      |
| Segvit-prune | Vit-large | 62.7 |  224.3 | [config](./config/prune/prune_segvit_pc.py)       |      |

### COCO-Stuff-10K
| Method       | Backbone  | mIoU | GFlops | config | ckpt |
|--------------|-----------|------|--------|--------|------|
| Segvit       | Vit-large | 47.4 |  366.9 | [config](./config/prune/BASE_segvit_cocostuff10k.py)       |      |
| Segvit-prune | Vit-large | 47.1 |  276.2 | [config](./config/prune/prune_segvit_cocostuff10k.py)       |      |



## License
For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors.

## Citation
