#!/bin/bash

python -u run.py \
    --model Edmot \
    --debug \
    --dataset Cora \
    --components 2 \
    --cutoff 50 \
    --task_name communityDetection \
    --epochs 1 \


#参数意义：
#    components
#    cutoff


# --model Edmot --debug --dataset Cora --components 2 --cutoff 50 --task_name communityDetection