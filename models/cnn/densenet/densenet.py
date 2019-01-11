import tensorflow as tf
import models.ops as ops
from models import debug


class DenseNet:
    """
    DenseNet-BC-121(k=12)
    """
    def __init__(self, blocks=(6, 12, 24, 16), growth_rates=4*(12,), compression=0.5, reuse=None,
                 classes=1, downsample=ops.avg_pool_2, activation=tf.nn.relu, keep_prob=None):
        self.training = None  # training_mode placeholder
        self.growth_rates = growth_rates
        self.compression = compression
        self.blocks = blocks
        self.classes = classes
        self.activation = activation
        self.downsample = downsample
        self.keep_prob = keep_prob
        self.reuse = reuse

    '''
    Following 4 functions define the Dense basic block.

    Composite function: 
        - Conv[ ReLU(BN(X)); 3, K, 1 ]
          Conv[ Input; kernel, features(channels), Stride ]
        - K = growth rate.
        - Batch-Norm -> ReLU -> Conv.
        - Dropout if self.keep_prob is not defined.    

    Bottleneck function:
        - Belongs to DenseNet-B.
        - Batch-Norm -> ReLU -> Conv(Kernel=1, Channels=4*Growth rate).

    Layer function: Basic layer of a Dense Block.
        - Bottleneck + Composite(Kernel=3, Channels=Growth rate)
        - Concatenate X and output of previous.

    Block function:
        - Builds a Dense Block with 'layers' layers and the specified growth rate.
    '''
    
    def composite(self, x, size, filters):
        bn = ops.batch_norm(x, self.training)
        a = self.activation(bn)
        out = ops.conv(a, size, filters, 1)
        if self.keep_prob is not None:
            keep_prob = tf.cond(self.training, lambda: self.keep_prob, lambda: 1.0)
            out = tf.nn.dropout(out, keep_prob)
        return out

    def bottleneck(self, x, growth_rate):
        return self.composite(x, 1, 4 * growth_rate)

    def layer(self, x, growth_rate):
        if debug:
            print(x)
        with tf.variable_scope('Bottleneck'):
            reduced = self.bottleneck(x, growth_rate)
        with tf.variable_scope('Composite'):
            new = self.composite(reduced, 3, growth_rate)
        output = tf.concat([x, new], 3)
        return output

    def block(self, x, layers, growth_rate):
        for i in range(layers):
            with tf.variable_scope('Layer_%d' % i):
                x = self.layer(x, growth_rate)
        return x


    '''
    Transition & Classification blocks.

    Transition function:
        - Defined the transtition block.
        - Compression Theta -> Downsampling.
        - Default compression=0.5.

    Classification function:
        - Final layers for classification task.
        - Avg. Pooling -> BN -> Activation -> Flatten -> Fully Connected.

    '''
    def transition(self, x, **kwargs):
        # Compression layer of 1x1 convolution.
        if self.compression != 1:
            if debug:
                print(x)
            with tf.variable_scope('Compression'):
                channels = x.get_shape()[-1].value
                filters = round(self.compression * channels)
                x = self.composite(x, 1, filters)
        if debug:
            print(x)
        with tf.variable_scope('Downsampling'):
            output = self.downsample(x, **kwargs)
        return output

    def classification(self, x):
        if debug:
            print(x)
        p = ops.global_avg_pool(x)
        if debug:
            print(p)
        bn = ops.batch_norm(p, self.training)
        a = self.activation(bn)
        channels = a.get_shape()[-1].value
        matrix = tf.reshape(a, [-1, channels])
        output = ops.dense(matrix, self.classes)
        return output

    '''
    Build Graph based on: Defaults:
        1. Growth rates list: growth_rates = 4*(12,) = (12, 12, 12, 12).
        2. Blocks list: blocks = ( 6, 12, 24, 16)    = ( 6, 12, 24, 16).
    '''
    def build_graph(self, x, training):
        if debug:
            print(x)
        # Training flag for Batch Normalization.
        self.training = training
        with tf.variable_scope('DenseNet', reuse=self.reuse):

            # TODO: According to DenseNet paper, each 'Conv' Stage: BN-ReLU-Conv.
            # Conv[Input; kernel, features(channels), Stride]
            x = ops.conv(x, 7, 2 * self.growth_rates[0], 2)
            x = ops.batch_norm(x, self.training)
            x = self.activation(x)
            if debug:
                print(x)
            x = ops.max_pool(x, 3, 2)

            # Main body of the DenseNet.
            # For each element in blocks put a Dense Block and Transtion Block.
            # Blocks = ( 6, 12, 24, 16).
            # Growth = (12, 12, 12, 12).
            for i, layers in enumerate(self.blocks):

                # Dense Block.
                with tf.variable_scope('Block_%d' % i):
                    x = self.block(x, layers, self.growth_rates[i])

                # Transition or classification block.
                if i < len(self.blocks) - 1:
                    with tf.variable_scope('Transition_%d' % i):
                        x = self.transition(x)
                else:
                    with tf.variable_scope('Classification'):
                        x = self.classification(x)
        return x


if __name__ == '__main__':
    print(DenseNet().build_graph(tf.ones([50, 224, 224, 3]), tf.constant(True)))
