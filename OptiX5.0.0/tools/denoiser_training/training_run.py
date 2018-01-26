#
# Copyright 2017 NVIDIA Corporation.  All rights reserved.
#
# This python script demonstrates how to use a deep learning network for denoising
# of monte-carlo rendered rgb images.
# It requires pairs of noisy and noise-free target images. The network will learn
# to remove noise from images.
# In order to improve the learning, additional inputs can be given, such as
# albedo and normal vector images.

import os
import sys
import glob
import thread_utils
import math
import multiprocessing
import getopt
import time
import numpy as np
import tensorflow as tf
np.random.seed(1234)
tf.set_random_seed(1234)



# Set these variables if TensorFlow should use a certain GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#avoid SSE warnings

###############################################################################
#
# Denoiser configuration
#
###############################################################################

# Input feature dimension
# For rgb training, set it to 3
# For rgb+albedo, set it to 6
# For rgb+albedo+normals, set it to 9
INPUT_DIMENSION = 3

# Number of training epochs. Each epoch is a complete pass over the training images
num_epochs = 20

# Save training data to a checkpoint file after each x epochs
save_after_num_epochs = 10

# Minibatch size. After each sequence of minibatch images the loss function is
# evaluated and weights are updated. Changing the value has an effect on the
# loss calculation as well on the performance.
minibatch_size = 4

# Crop image size. During training a random image region is used for comparison
# of input and target.
cropSize = [256, 256]

# Path where graph and tensorboard statistic files are saved. The graph file is
# needed when generating the weights for inference.
SAVE_DIRECTORY = "training_result"

# Configuration of training image files
# TRAIN_DIRECTORY contains three subdirectories:
# rgb
#   This directory contains rgb images for training. The filenames must end
#   with _XXXXXX.npy where XXXXXX is the number of samples used in rendering.
#   For each scene there can be multiple such progressive rendered images. The target
#   (ground thruth / reference) image filename must end with _target.npy for each
#   set of iteration images.
# albedo
#   Contains albedo/bsdf images
# normals
#   Contains normal/xyz images

TRAIN_DIRECTORY  = 'training_data'

###############################################################################

restore = None

def print_usage():
    print '\033[93m'
    print sys.argv[0], '[--help] [--dimension {3,6,9}] [--save e] [--train dir] [--result dir] [--epochs e] [--restore checkpointfile]'
    print 'defaults: dimension=3, save=10, train=training_data, result=training_result, epochs=20, restore=none'
    print '\033[0m'

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "h", ["help","dimension=","save=","train=","result=","epochs=","restore="])
except getopt.GetoptError:
    print_usage()
    sys.exit()

for opt, arg in opts:
    if opt == '-h' or opt in("--help"):
	print_usage()
	sys.exit()
    elif opt in ("--dimension"):
	INPUT_DIMENSION = int(arg)
    elif opt in ("--save"):
	save_after_num_epochs = int(arg)
    elif opt in ("--train"):
	TRAIN_DIRECTORY = arg
    elif opt in ("--result"):
	SAVE_DIRECTORY = arg
    elif opt in ("--epochs"):
	num_epochs = int(arg)
    elif opt in ("--restore"):
	restore = arg

if (not os.path.isdir(TRAIN_DIRECTORY)):
    print '\033[31m', 'Train directory %s does not exist' % TRAIN_DIRECTORY, '\033[0m'
    sys.exit()

if (not os.path.isdir(SAVE_DIRECTORY)):
    print '\033[31m', 'Save directory %s does not exist' % SAVE_DIRECTORY, '\033[0m'
    sys.exit()

###############################################################################
#
# Autoencoder network definition
#
###############################################################################

def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope(name):
		#mean = tf.reduce_mean(var)
		#tf.summary.scalar('mean', mean)
		#with tf.name_scope('stddev'):
		#	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		#tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def conv2d(x, out_channel, name='conv', relu = True):

	with tf.name_scope(name) as scope:
		in_shape = x.get_shape().as_list()
		in_channel = in_shape[-1]

		# Weights according to He initializer
		filter_shape = [3, 3, in_channel, out_channel]

		fan_in = 3.0*3.0*in_channel
		if relu:
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=math.sqrt(2.0 / fan_in)), name='W')
			variable_summaries(W, 'weight')
		else:
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=math.sqrt(1.0 / fan_in)), name='W')
			variable_summaries(W, 'weight')
		b = tf.Variable(tf.zeros([out_channel]), name='b')

		res = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
		res = tf.nn.bias_add(res, b ,name='bias')
		if relu:
			res = tf.nn.relu(res, name = 'relu')

	return res

def maxpool2d(x, k=2, name='pool'):
	# MaxPool2D wrapper
	with tf.name_scope(name) as scope:
		res = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
							  padding='SAME', name=name)
	return res

def unpool(value, name='unpool'):
	with tf.name_scope(name) as scope:
		in_shape = value.get_shape().as_list()
		in_channel = in_shape[-1]

		filter_shape = [2, 2, in_channel, in_channel]
		out_shape = [in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]]

		# Setup weights for nearest neighbour upscaling
		weights = np.zeros(filter_shape)
		for y in range(0, filter_shape[0]):
			for x in range(0, filter_shape[1]):
				for ic in range(0, filter_shape[2]):
					for oc in range(0, filter_shape[3]):
						if ic == oc:
							weights[y][x][ic][oc] = 1.0
						else:
							weights[y][x][ic][oc] = 0.0
		W = tf.constant(weights, dtype=tf.float32, shape=filter_shape)
		res = tf.nn.conv2d_transpose(value, W, output_shape=out_shape, strides=[1, 2, 2, 1], padding='SAME')
		return tf.nn.relu(res)

def unpool_concat(a, b, name='upconcat'):
	with tf.name_scope(name) as scope:
		up = unpool(a)
		res = tf.concat([up, b], 3)
	return res

def concat(a, b, name):
	with tf.name_scope(name) as scope:
		return tf.concat([a, b], 3)

def autoencoder(x):

	prevLayer = conv1  = conv2d(x, 32, 'conv1')
	prevLayer = conv1b = conv2d(prevLayer, 32, 'conv1b')
	prevLayer = pool1  = maxpool2d(prevLayer, 2, 'pool1') # 256 -> 128

	prevLayer = conv2 = conv2d(prevLayer, 48, 'conv2')
	prevLayer = pool2 = maxpool2d(prevLayer, 2, 'pool2') # 128 -> 64

	prevLayer = conv3 = conv2d(prevLayer, 56, 'conv3')
	prevLayer = pool3 = maxpool2d(prevLayer, 2, 'pool3') # 64 -> 32

	prevLayer = conv4 = conv2d(prevLayer, 80, 'conv4')
	prevLayer = pool4 = maxpool2d(prevLayer, 2, 'pool4') # 32 -> 16

	prevLayer = conv5 = conv2d(prevLayer,  104, 'conv5')
	prevLayer = pool5 = maxpool2d(prevLayer, 2, 'pool5') # 16 -> 8

	prevLayer = us6 = unpool_concat(prevLayer, pool4, 'unpool4')
	prevLayer = conv6 = conv2d(prevLayer,  152, 'conv6')
	prevLayer = conv6b = conv2d(prevLayer, 152, 'conv6b')

	prevLayer = us7 = unpool_concat(prevLayer, pool3, 'unpool3')
	prevLayer = conv7 = conv2d(prevLayer, 112, 'conv7')
	prevLayer = conv7b = conv2d(prevLayer, 112, 'conv7b')

	prevLayer = us8 = unpool_concat(prevLayer, pool2, 'unpool2')
	prevLayer = conv8 = conv2d(prevLayer, 88, 'conv8')
	prevLayer = conv8b = conv2d(prevLayer, 88, 'conv8b')

	prevLayer = us9 = unpool_concat(prevLayer, pool1, 'unpool1')
	prevLayer = conv9 = conv2d(prevLayer,  64, 'conv9')
	prevLayer = conv9b = conv2d(prevLayer, 64, 'conv9b')

	prevLayer = us10 = unpool_concat(prevLayer, x, 'unpool0')	
	prevLayer = conv10 = conv2d(prevLayer, 64, 'conv10')
	prevLayer = conv10b = conv2d(prevLayer, 32, 'conv10b')

	out = conv2d(prevLayer, 3, 'conv11', relu = False)

	#print("Output var: %s" % out.name)

	return out

###############################################################################
#
# Loss calculations
#
###############################################################################

AEInputImg  = tf.placeholder(tf.float32, shape=[minibatch_size, cropSize[0], cropSize[1], INPUT_DIMENSION], name="Input")
AETargetImg = tf.placeholder(tf.float32, shape=[minibatch_size, cropSize[0], cropSize[1], 3], name="Target")
AELearningRate = tf.placeholder(tf.float32, shape=[])

AEOutput = autoencoder(AEInputImg)

# L2 loss function
with tf.name_scope("loss") as scope:
	AECost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(AETargetImg, AEOutput))))
tf.summary.scalar('loss', AECost)

AEOptimizer = tf.train.AdamOptimizer(learning_rate=AELearningRate).minimize(AECost)

init = tf.global_variables_initializer()

# Configuration of learning rate decay
learning_rate_max     = 0.001
learning_rate_initial = learning_rate_max / 10
learning_rate_rampup_length = 10
learning_rate_rampup  = (learning_rate_max/learning_rate_initial) ** (1./learning_rate_rampup_length)
learning_rate         = learning_rate_initial
learning_rate_den     = 1000 // minibatch_size

# Add ops to save and restore all the variables.
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
merged = tf.summary.merge_all()

###############################################################################
# 
# Crop image generation from training data for minibatches
#
###############################################################################

# return shape of a npy file (i.e. resolution)
def getshape(file):
    rd = open(file, 'rb')
    data = rd.read(80)
    x = data.split()
    return [int(x[5][1:-1]), int(x[6][0:-1]), int(x[7][0:-2])]

# Utility class to handle reading input images
class TrainingInput:

	def __init__(self, TDIR):

		self.train_dir = TDIR
		# get all training images (i.e. exclude all _target.npy images)
		self.trainFiles = [fn for fn in glob.glob(os.path.join(self.train_dir+"/rgb", '*.npy'))
				   if not os.path.basename(fn).endswith('_target.npy')]
		print 'Number of training files:', len(self.trainFiles) 
		if (len(self.trainFiles) == 0):
		    print ('No training files')
		    sys.exit()

	def iterate_minibatches(self, batchsize, epoch, dim, shuffle=False, cropsize=None):
		# limit number of files to multiples of batchsize, which is a power of 2
		num = len(self.trainFiles) & ~(batchsize-1)
		indices = np.arange(num) 
		if shuffle: 
			np.random.shuffle(indices) 

		for start_idx in range(0, num, batchsize):
			start_idx = start_idx + epoch
			excerpt = indices[start_idx: min(start_idx + batchsize,num)]                # indices for this minibatch

			inputs  = []
			targets = []

			for i in excerpt:
				input_file  = self.trainFiles[i]
				target_file = os.path.join(self.train_dir+"/rgb", os.path.basename(input_file)[:-10] + "target.npy")
				albedo_file = os.path.join(self.train_dir+"/albedo", os.path.basename(input_file)[:-4] + ".npy")
				normal_file = os.path.join(self.train_dir+"/normal", os.path.basename(input_file)[:-4] + ".npy")

				# source image size
				sh = getshape(input_file)
				w = sh[1]
				h = sh[0]

				# Select a random crop
				cw, ch = w, h
				ow, oh = 0,0
				if cropsize != None:
					cw,ch = cropsize[1],cropsize[0]
					sw,sh = max(0,w-cw), max(0,h-ch)
					ow,oh = np.random.randint(0,sw+1), np.random.randint(0,sh+1)

				imgs = [np.memmap(input_file,  dtype='float32', mode='r', offset=80+oh*w*3, shape=(ch, w, 3))[:,ow:ow+cw,:]]
				if (dim >= 6):
				    imgs.append(np.memmap(albedo_file,  dtype='float32', mode='r', offset=80+oh*w*3, shape=(ch, w, 3))[:,ow:ow+cw,:])
				if (dim == 9):
				    imgs.append(np.memmap(normal_file,  dtype='float32', mode='r', offset=80+oh*w*3, shape=(ch, w, 3))[:,ow:ow+cw,:])
				imgs.append(np.memmap(target_file,  dtype='float32', mode='r', offset=80+oh*w*3, shape=(ch, w, 3))[:,ow:ow+cw,:])

				imgs = [np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)  for img in imgs]

				# Slice and dice.
				if (dim == 3):
				    inputs.append(imgs[0])
				    targets.append(imgs[1]) # target
				if (dim == 6):
				    inputs.append(np.concatenate((imgs[0], imgs[1]))) # input + features
				    targets.append(imgs[2]) # target
				if (dim == 9):
				    inputs.append(np.concatenate((imgs[0], imgs[1], imgs[2]))) # input + features
				    targets.append(imgs[3]) # target

			yield inputs, targets

	def getNumElements(self):
		return len(self.trainFiles)

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = unichr(0x78) * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

def shuffleImg(x):
        x = np.swapaxes(x, 1, 2);
        x = np.swapaxes(x, 2, 3);
        return x;

###############################################################################
#
# training
#
###############################################################################

with tf.Session() as sess:

	if (restore != None):
	    saver.restore(sess, restore)
	    print ("Restored model", restore)
	else:
	    sess.run(init)
	
	np.random.seed(1234)
	tf.set_random_seed(1234)


	print '\033[92m'
	print 'Start training: dim=%d, train dir=%s, save dir=%s, epochs=%d' % (INPUT_DIMENSION, TRAIN_DIRECTORY, SAVE_DIRECTORY, num_epochs)
	print '\033[0m'

	trainClass = TrainingInput(TRAIN_DIRECTORY)
	train_writer = tf.summary.FileWriter(SAVE_DIRECTORY+"/run", sess.graph)

	numMiniBatchesProcessed = 0
	trainElements = trainClass.getNumElements() / minibatch_size
	avgLossTraining = 1.0

	tf.train.write_graph(sess.graph_def, SAVE_DIRECTORY, "graph.pb", True)

	step = 0
	for epoch in range(1, num_epochs+1):

		start_time = time.time()
		printProgress(0, trainElements, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
		with thread_utils.ThreadPool(multiprocessing.cpu_count()) as thread_pool:     
			minibatches = trainClass.iterate_minibatches(minibatch_size, 0, INPUT_DIMENSION, shuffle=True, cropsize=cropSize)
			minibatches = thread_utils.run_iterator_concurrently(minibatches, thread_pool)
			sum = 0.0
			num = 0.0
			for batch in minibatches:
				inputs, targets = batch 

				inputs = shuffleImg(inputs)
				targets = shuffleImg(targets)

				[summary, cost, dummy] = sess.run([merged, AECost, AEOptimizer], feed_dict={AEInputImg: inputs, AETargetImg: targets, AELearningRate: learning_rate})
				sum += cost

				numMiniBatchesProcessed += 1
				num += 1

				if num == trainElements -1:
					train_writer.add_summary(summary, step)
					step += 1

				printProgress(num, trainElements, prefix = 'Progress:', suffix = 'Complete', barLength = 50)

				# Learning rate rampup.
				if numMiniBatchesProcessed <= learning_rate_rampup_length:
					learning_rate *= learning_rate_rampup

				# Reduce learning rate.
				if numMiniBatchesProcessed >= learning_rate_den:
					n = numMiniBatchesProcessed // learning_rate_den
					learning_rate = learning_rate_max / (n**.5)         # As decribed in the ADAM paper

		duration = time.time() - start_time
		if not num == 0:
			avgLossTraining = sum/num
		else:
			avgLossTraining = sum
		remaining = (num_epochs-epoch)*duration/(60*60)
		timestring = "hours"
		if (remaining < 1):
		    remaining *= 60
		    timestring = "minutes"
		if (remaining < 1):
		    remaining *= 60
		    timestring = "seconds"
		print("Epoch %3d - Learn rate: %1.6f - train loss: %5.5f - time %.1f ms (remaining %.1f %s)" % (epoch, learning_rate, avgLossTraining, duration*1000.0, remaining, timestring))

		if epoch % save_after_num_epochs == 0 or epoch == num_epochs:
			save_path = saver.save(sess, SAVE_DIRECTORY + "/model_%d.ckpt" % epoch)
