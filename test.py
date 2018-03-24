import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
# import tensorflow.contrib.slim.nets

from vgg import *
from model import *

batch_size = 16
height = 224
width = 224
channels = 3

X1 = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels])
X2 = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels])
Y = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels])
Y_map = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels])

left = np.ones( (batch_size, height, width, channels), dtype=np.float32 )
right = np.ones( (batch_size, height, width, channels), dtype=np.float32 )
disp = np.ones( (batch_size, height, width, channels), dtype=np.float32 )
out_map = np.ones( (batch_size, height, width, channels), dtype=np.float32 )

# vgg = tf.contrib.slim.nets.vgg

# with slim.arg_scope( vgg_arg_scope() ):
# 	out = vgg_16( images, is_training=False )

arg = {}
net = model(arg)
prediction = net( X1, X2, is_train=True )

loss_op = tf.losses.mean_squared_error( Y, Y_map*prediction )
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

all_vars = tf.all_variables()
trainable_vars = tf.trainable_variables()
other_vars = list( set(all_vars) - set(trainable_vars) )

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
	train_op = optimizer.minimize(loss_op)


init = tf.global_variables_initializer()

# from tensorflow.python.tools import inspect_checkpoint as chkp

# # print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file("vgg_16.ckpt", tensor_name='', all_tensors=True)


with tf.Session() as sess:
	
	sess.run(init)
	
	saver = tf.train.Saver( other_vars )
	saver.restore(sess, 'vgg_16.ckpt')
		
	# pass images through the network
	fc6_output = sess.run( prediction, feed_dict={X1:left, X2:right} )
	print( fc6_output.shape  )
