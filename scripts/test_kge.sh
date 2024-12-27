#!/bin/bash

NODE_INIT_METHOD="gcl"       # gcl, lm, random
PRETRAINED_PATH=""              # Path to kge model ends with .ckpt

# Set INIT_DIM based on NODE_INIT_METHOD
if [[ "$NODE_INIT_METHOD" == "random" || "$NODE_INIT_METHOD" == "lm" ]]; then
    INIT_DIM=768
else
    INIT_DIM=256
fi

# Load from gcl checkpoint
GCL_MODEL="ggd"                 # dgi, grace, ggd (use if the node_init_method is gcl)
GCL_FUSE_METHOD="attention"     # attention, redaf, none (use if the node_init_method is gcl)

NEG_RATIO=3
BATCH_SIZE=64
DEVICES="[0]"

# Run the Python training script with specified parameters
python3 test_kge.py \
    devices=$DEVICES \
    neg_ratio=$NEG_RATIO \
    gcl_model=$GCL_MODEL \
    gcl_fuse_method=$GCL_FUSE_METHOD \
    data.batch_size=$BATCH_SIZE \
    data.embed_dim=$INIT_DIM \
    data.node_init_method=$NODE_INIT_METHOD \
    pretrained_path=$PRETRAINED_PATH
