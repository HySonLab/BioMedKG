#!/bin/bash

ENCODER="rgcn"                  # rgcn, rgat
DECODER="dismult"               # transe, dismult, complex
NODE_INIT_METHOD="random"       # gcl, lm, random
PRETRAINED_PATH=""              # Path to kge model ends with .ckpt

# Set INIT_DIM based on NODE_INIT_METHOD
if [[ "$NODE_INIT_METHOD" == "random" || "$NODE_INIT_METHOD" == "lm" ]]; then
    INIT_DIM=768
else
    INIT_DIM=256
fi

FUSE_METHOD="none"              # attention, redaf, none (use if the node_init_method is lm)
# Load from gcl checkpoint
GCL_MODEL="ggd"                 # dgi, grace, ggd (use if the node_init_method is gcl)
GCL_FUSE_METHOD="attention"     # attention, redaf, none (use if the node_init_method is gcl)

EPOCHS=1
NEG_RATIO=10
BATCH_SIZE=8
DEVICES="[0]"
LEARNING_RATE=0.001

# Run the Python training script with specified parameters
python3 train_dpi.py \
    devices=$DEVICES \
    epochs=$EPOCHS \
    neg_ratio=$NEG_RATIO \
    gcl_model=$GCL_MODEL \
    gcl_fuse_method=$GCL_FUSE_METHOD \
    data.batch_size=$BATCH_SIZE \
    data.embed_dim=$INIT_DIM \
    data.node_init_method=$NODE_INIT_METHOD \
    model.in_dim=$INIT_DIM \
    model.learning_rate=$LEARNING_RATE \
    model.fuse_method=$FUSE_METHOD \
    model.encoder_name=$ENCODER \
    model.decoder_name=$DECODER \
    pretrained_path=$PRETRAINED_PATH    \
    debug=true
