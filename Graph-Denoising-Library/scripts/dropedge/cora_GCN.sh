#!/bin/bash

python -u run.py \
    --model DropEdge \
    --debug \
    --datapath data/cora \
    --seed 42 \
    --dataset cora \
    --model_type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 2 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.01 \
    --weight_decay 0.005 \
    --early_stopping 400 \
    --sampling_percent 0.7 \
    --dropout 0.8 \
    --use_gpu false \
    --normalization FirstOrderGCN 
    
