#!/bin/bash

NODE_TYPE="gene"                # gene, drug, disease
MODEL_NAME="ggd"                # dgi, grace, ggd
FUSE_METHOD="attention"         # attention, redaf, none

LR=0.001
BATCH_SIZE=64
DEVICES="[0]"
EPOCHS=100
NODE_INIT_METHOD="lm"           # lm, random

python3 train_gcl.py \
    data.node_type=$NODE_TYPE \
    data.node_init_method=$NODE_INIT_METHOD \
    data.batch_size=$BATCH_SIZE \
    model.model_name=$MODEL_NAME \
    model.learning_rate=$LR \
    model.fuse_method=$FUSE_METHOD \
    devices=$DEVICES \
    epochs=$EPOCHS  \
    debug=true
