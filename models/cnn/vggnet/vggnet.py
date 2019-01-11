import tensorflow as tf
import model.ops as ops
from model import debug


class VggNet:
    """
    VggNet-11(A): vgg_layer_repetitions=(1, 1, 2, 2, 2)
    VggNet-13(B): vgg_layer_repetitions=(2, 2, 2, 2, 2)
    VggNet-16(D): vgg_layer_repetitions=(2, 2, 3, 3, 3)
    """
    def __init__(self,
                 vgg_layer_filters=(64, 128, 256, 512, 512),
                 vgg_layer_repetitions=(1, 1, 2, 2, 2),
                 dense_layer_filters=(4096, 4096),
                 classes=2, use_bn=False, keep_prob=None):
        self.vgg_layer_filters = vgg_layer_filters
        self.vgg_layer_repetitions = vgg_layer_repetitions
        self.dense_layer_filters = dense_layer_filters
        self.classes = classes
        self.use_bn = use_bn
        self.keep_prob = keep_prob
        self.training = None

    def vgg_layer(self, x, filters, repetitions=2):
        for i in range(repetitions):
            with tf.variable_scope('sublayer_%d' % i):
                if debug:
                    print(x)
                x = ops.conv(x, 3, filters)
                if self.use_bn:
                    x = ops.batch_norm(x, self.training)
                x = tf.nn.relu(x)
        if debug:
            print(x)
        x = ops.max_pool(x, 2)
        return x

    def build_graph(self, x, training):
        self.training = training
        with tf.variable_scope('VggNet'):
            for i, layer in enumerate(zip(self.vgg_layer_filters, self.vgg_layer_repetitions)):
                filters, repetitions = layer
                with tf.variable_scope('VggLayer_%d' % i):
                    x = self.vgg_layer(x, filters, repetitions)
            b, h, w, d = tuple(map(lambda dim: dim.value, x.get_shape()))
            if debug:
                print(x)
            x = tf.reshape(x, [-1, h * w * d])
            for i, filters in enumerate(self.dense_layer_filters):
                with tf.variable_scope('DenseLayer_%d' % i):
                    if debug:
                        print(x)
                    x = ops.dense(x, filters, use_bias=not self.use_bn)
                    if self.use_bn:
                        x = ops.batch_norm(x, self.training)
                    x = tf.nn.relu(x)
                    if self.keep_prob is not None:
                        keep_prob = tf.cond(self.training, lambda: self.keep_prob, lambda: 1.0)
                        x = tf.nn.dropout(x, keep_prob)
            with tf.variable_scope('Classification'):
                if debug:
                    print(x)
                x = ops.dense(x, self.classes)
        return x


if __name__ == '__main__':
    print(VggNet().build_graph(tf.ones([50, 224, 224, 3]), tf.constant(False)))
