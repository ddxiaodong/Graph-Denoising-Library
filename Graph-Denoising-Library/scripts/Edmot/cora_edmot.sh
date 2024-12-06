#!/bin/bash

python -u run.py \
    --model Edmot \
    --debug \
    --dataset Cora \
    --components 2 \
    --cutoff 50 \
    --task_name communityDetection \
