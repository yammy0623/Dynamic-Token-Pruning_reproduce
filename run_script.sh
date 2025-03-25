#!/bin/bash
# Reproduce Ade20k
# ./tools/dist_train.sh config/prune/BASE_segvit_ade20k.py ./exp/BASE_segvit_ade20k
# ./tools/dist_train_load.sh config/prune/prune_segvit_ade20k.py ./exp/PRUNE_segvit_ade20k ./exp/BASE_segvit_ade20k/iter_40000.pth
# ./tools/dist_test.sh  config/prune/prune_segvit_ade20k.py ./exp/PRUNE_segvit_ade20k/iter_40000.pth 2
# ./tools/dist_test.sh  config/prune/prune_segvit_ade20k.py ./exp/BASE_segvit_ade20k/iter_40000.pth 2 >> ./exp/Ade20k_test_result.txt

# Reproduce PascalContext
# ./tools/dist_train.sh config/prune/BASE_segvit_pc.py ./exp/BASE_segvit_pc
# ./tools/dist_train_load.sh config/prune/prune_segvit_ade20k.py ./exp/PRUNE_segvit_pc ./exp/BASE_segvit_pc/iter_40000.pth


# Reproduce COCO
# ./tools/dist_train.sh config/prune/BASE_segvit_cocostuff10k.py ./exp/BASE_segvit_coco
# ./tools/dist_train_load.sh config/prune/prune_segvit_cocostuff10k.py ./exp/PRUNE_segvit_coco ./exp/BASE_segvit_coco/iter_64000.pth
./tools/dist_test.sh config/prune/prune_segvit_cocostuff10k.py ./exp/BASE_segvit_coco/iter_64000.pth 2 >> ./exp/COCO_test_result.txt
./tools/dist_test.sh config/prune/prune_segvit_cocostuff10k.py ./exp/PRUNE_segvit_coco/iter_20000.pth 2 >> ./exp/COCO_test_result.txt
