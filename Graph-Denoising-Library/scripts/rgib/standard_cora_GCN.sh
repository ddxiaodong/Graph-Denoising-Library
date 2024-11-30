#!/bin/bash


python3 -u run.py \
      --gnn_model GCN \
      --num_gnn_layers 4 \
      --dataset Cora \
      --noise_ratio 0.2 \

standard-training.py --gnn_model GCN  --num_gnn_layers 4 --dataset Cora --noise_ratio 0.2
