#!/bin/bash

python -u run.py \
    --model RGIB \
    --gnn_model GCN \
    --num_gnn_layers 4 \
    --dataset Cora \
    --noise_ratio 0.2 \
    --scheduler linear \
    --scheduler_param 1.0 \
    --task_name linkPrediction \
