import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

from vgg import *

class model(object):
	
	"""docstring for model"""
	def __init__(self, arg):
		self.arg = arg


	def __call__(self, left, right, is_train):

		with slim.arg_scope( vgg_arg_scope() ):
			left_emb = vgg_16( left, is_training=False )

		with slim.arg_scope( vgg_arg_scope() ):
			right_emb = vgg_16( right, is_training=False )


		self.net = tf.concat( [left_emb, right_emb], axis=3 )

		with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:
			self.net = tf.layers.conv2d_transpose( self.net , filters=512, kernel_size=[2,2], strides=2, name="deconv1", reuse=tf.AUTO_REUSE, trainable=is_train )
			self.net = tf.layers.conv2d_transpose( self.net , filters=512, kernel_size=[2,2], strides=2, name="deconv2", reuse=tf.AUTO_REUSE, trainable=is_train )
			self.net = tf.layers.conv2d_transpose( self.net , filters=256, kernel_size=[2,2], strides=2, name="deconv3", reuse=tf.AUTO_REUSE, trainable=is_train )
			self.net = tf.layers.conv2d_transpose( self.net , filters=128, kernel_size=[2,2], strides=2, name="deconv4", reuse=tf.AUTO_REUSE, trainable=is_train )
			self.net = tf.layers.conv2d_transpose( self.net , filters=64, kernel_size=[2,2], strides=2, name="deconv5", reuse=tf.AUTO_REUSE, trainable=is_train )
			self.net = tf.layers.conv2d( self.net , filters=1, kernel_size=[1,1], strides=1, name="pred", reuse=tf.AUTO_REUSE, trainable=is_train )
		
		return( self.net )
