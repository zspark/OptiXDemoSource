from __future__ import print_function

import glob
import os
import sys
import math
import time
import struct

import numpy as np

import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import attr_value_pb2

graph_def = graph_pb2.GraphDef()

input_binary = True

def parse_weight(node):
	tp = node.attr['value'].tensor # TensorProto objects

	byte_values = tp.tensor_content
	blen = len(byte_values)
	W = np.frombuffer(byte_values, dtype=np.float32)

	# Tensor dims 
	dims = np.zeros(4, 'int32')

	dims[0] = tp.tensor_shape.dim[0].size #h
	dims[1] = tp.tensor_shape.dim[1].size #w
	dims[2] = tp.tensor_shape.dim[2].size #cin
	dims[3] = tp.tensor_shape.dim[3].size #cout

	W = np.reshape(W, [dims[0],dims[1], dims[2], dims[3]])

	# reshuffle w_dims to align with cuDNN
	dims[0] = tp.tensor_shape.dim[3].size #cout
	dims[1] = tp.tensor_shape.dim[2].size #cin
	dims[2] = tp.tensor_shape.dim[1].size #w
	dims[3] = tp.tensor_shape.dim[0].size #h

	# Reshuffle data to align with order in cuDNN...
	Wp = np.zeros([dims[0],dims[1], dims[2], dims[3]],dtype=np.float32)
	for oc in range(dims[0]):
		for ic in range(dims[1]):
			for x in range(dims[2]):
				for y in range(dims[3]):
					Wp[oc,ic,x,y] = W[y,x, ic, oc]

	wr = np.ravel(Wp, order='C')
	return dims, wr

def parse_bias(node):
	tp = node.attr['value'].tensor # TensorProto objects

	byte_values = tp.tensor_content
	blen = len(byte_values)
	float_values = np.frombuffer(byte_values, dtype=np.float32)
	#float_values = np.zeros(blen, dtype=np.float32)
	cout = tp.tensor_shape.dim[0].size
	return cout, float_values

conv_layers = ['conv1', 'conv1b', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv6b',
			   'conv7', 'conv7b', 'conv8', 'conv8b', 'conv9', 'conv9b', 'conv10', 'conv10b',
			   'conv11'] 

def dump_layer(output_file, layer, layerID, found_variables):
	w_node = found_variables[l+'/W']
	b_node = found_variables[l+'/b']

	w_dims, wr = parse_weight(w_node)
	b_dims, b  = parse_bias(b_node)

	# Build binary blob
	output_file.write(struct.pack("i", w_dims[0]))
	output_file.write(struct.pack("i", w_dims[1]))
	output_file.write(struct.pack("i", w_dims[2]))
	output_file.write(struct.pack("i", w_dims[3]))
	
	blob = np.concatenate((wr, b)).astype(np.float32)

	blob.tofile(output_file)

	print(layerID, l, blob.shape, w_dims, b_dims)

if len(sys.argv) < 3:
	print("Usage: python parse_tf_network.py ./network.pb training.bin")
	print("First freeze the network using:")
	print("""python /usr/local/lib/python2.7/dist-packages/tensorflow/python/tools/freeze_graph.py --input_graph="./graph.pb" --input_checkpoint="./model_10.ckpt" --output_graph="model_10.pb" --output_node_names="conv11/bias""")
	print("--------------------------------------------")
	sys.exit(1)

def read_db(file):
    f = open(file, "rb")
    graph_def.ParseFromString(f.read())
    # Build a dict with all the conv layer nodes and names
    variable_names = []
    nodes = []
    for node in graph_def.node:
	if node.name[:4] == "conv":
	  nodes.append(node)
	  variable_names.append(node.name)
    return dict(zip(variable_names, nodes))

nset = 1
if (len(sys.argv) >= 6):	# progn + outputfile + rgb.db + rgb_name + albedo.db + albedo_name
    nset += 1
if (len(sys.argv) == 8):	# progn + outputfile + rgb.db + rn + albedo.db + an + normal.db + nn 
    nset += 1

output_file = open(sys.argv[1], 'wb')
output_file.write(struct.pack("i", 1))		# version
output_file.write(struct.pack("i", 0))		# reserved
output_file.write(struct.pack("i", nset))	# number of training sets

param = 2
poslist = []
for x in range (nset):
    output_file.write(sys.argv[param+1].ljust(32, "\0"))
    poslist.append(output_file.tell())
    output_file.write(struct.pack("i", 0))	# offset to directory data
    param += 2

param = 2
for x in range (nset):
    print ("---> set:", sys.argv[param], "name:", sys.argv[param+1])
    vars = read_db(sys.argv[param])
    pos = output_file.tell()
    output_file.seek(poslist[x], 0)
    output_file.write(struct.pack("i", pos))
    output_file.seek(pos, 0)
    output_file.write(struct.pack("i", 17))		# 17 layers
    for l,layerID in zip(conv_layers, range(len(conv_layers))):
	    dump_layer(output_file, l, layerID, vars)
    param += 2

output_file.close()
print('wrote', sys.argv[1])
