#!/bin/bash

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2

if [ $# -ne 3 ]; then
    echo "use: genlayer.sh result-dir tensorflow_checkpoint_number {rgb, rgb-albedo or rgb-albedo-normal}"
    echo "example: genlayer.sh ../result 100 rgb-albedo will generate a ../result/training_100.bin training file for rgb-albedo input"
    exit 1
fi

#python /usr/local/lib/python2.7/dist-packages/tensorflow/python/tools/freeze_graph.py \

python ./freeze_graph.py \
--input_graph=$1/graph.pb \
--input_checkpoint=$1/model_$2.ckpt \
--output_graph=$1/model_$2.pb \
--output_node_names="conv11/bias"

# create training file:
python parse_network.py $1/training_$2.bin $1/model_$2.pb $3
