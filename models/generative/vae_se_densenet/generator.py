import tensorflow as tf
from models.cnn.densenet.densenet import DenseNet
from models.cnn.squeeze_excitation_net.se_net import squeeze_excitation_layer


class Generator(DenseNet):
    def __init__(self, blocks, growth_rates, final_filters, shape, downsample):
        super().__init__(blocks=blocks, growth_rates=growth_rates, downsample=downsample)
        self.final_filters = final_filters
        self.shape = shape

    '''
    Generator Graph:
        - Smaller than the DensetNet for classification default.
        - blocks= (4, 2) Simetric to Encoder
        - growth_rates=(8, 8)
        - downsample=ops.conv_t_5  
        - Final Graph:
            * Dense Block( 2 layers , 8 growth rate) --> Transition ( Compression Layer + Downsampling) 
            --> Dense Block( 2 layers , 8 growth rate) --> Transition ( Compression Layer + Downsampling)             
    '''
    # TODO: Why has he changed it to a pure convolutional instead avg pool? (Same as Encoder)
    def build_graph(self, x, training, reuse=None):
        x = tf.reshape(x, self.shape)
        self.training = training
        with tf.variable_scope('Generator', reuse=reuse):
            for i, layers in enumerate(self.blocks):
                # Dense block.
                with tf.variable_scope('Block_%d' % i):
                    x = self.block(x, layers, self.growth_rates[i])

                # Squeeze-Excitation block.
                with tf.variable_scope('SE_DenseBlock_%d' % i):
                    x = squeeze_excitation_layer(x)

                # Transition Block.
                with tf.variable_scope('Transition_%d' % i):
                    # Last transition block: final_filters=3
                    # Convolution transpose
                    filters = None if i < len(self.blocks) - 1 else self.final_filters
                    x = self.transition(x, filters=filters)

                # Squeeze-Excitation block.
                with tf.variable_scope('SE_Transition_%d' % i):
                    x = squeeze_excitation_layer(x)
        return tf.nn.sigmoid(x)


if __name__ == '__main__':
    print(Generator().build_graph(tf.ones([25, 56 * 56]), True))