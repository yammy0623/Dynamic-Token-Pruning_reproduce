#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
./tools/dist_train.sh config/prune/prune_segvit_ade20k.py ./exp