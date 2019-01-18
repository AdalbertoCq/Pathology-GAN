import tensorflow as tf

def leakyReLU(x, alpha):
    return tf.maximum(alpha*x, x)