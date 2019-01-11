import tensorflow as tf
import models.ops as ops
from models import debug


'''
Squeeze-Excitation function: 
    - Ratio compresses the channels in the NN. 
    - Global avg pooling for every channel.   (batch_size, 1, 1, C)
    - Compression given by ratio.             (batch_size, 1, 1, C/ratio)
    - Scale each channel as chosen by the NN. (batch_size, 1, 1, C)

'''

def squeeze_excitation_layer(x, ratio=16):
    channels = int(x.shape[-1])
    compress = round(channels/ratio)
    # Global Average Pooling.
    with tf.variable_scope('Squeeze'):
        squeeze = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        squeeze = tf.layers.flatten(squeeze)
    with tf.variable_scope('Excitation_1'):
        # Fully connected layer, implement with a conv2d.
        excitation = ops.dense(squeeze, units=compress,activation=tf.nn.relu)
    with tf.variable_scope('Excitation_2'):
        # Fully connected layer.
        excitation = ops.dense(excitation, units=channels,activation=tf.sigmoid)
    excitation = tf.reshape(excitation, [-1, 1, 1, channels])
        # Scale
    scaled = x * excitation
    return scaled

    