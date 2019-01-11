import tensorflow as tf
from models.cnn.densenet.densenet import DenseNet
from models.cnn.squeeze_excitation_net.se_net import squeeze_excitation_layer


class Encoder(DenseNet):
    def __init__(self, blocks, growth_rates, final_filters, shape, downsample, activation):
        # DenseNet Constructor
        super().__init__(blocks=blocks, growth_rates=growth_rates, downsample=downsample, activation=activation)

        # Last layer filers and encoding shape.
        self.final_filters = final_filters
        self.shape = shape

    '''
    Encoder Graph:
        - Smaller than the DensetNet for classification default.
        - blocks=(2, 4)
        - growth_rates=(8, 8)
        - downsample=ops.conv_5_2  
        - Final Graph:
            * Dense Block( 2 layers , 8 growth rate) --> Transition ( Compression Layer + Downsampling) 
            --> Dense Block( 2 layers , 8 growth rate) --> Transition ( Compression Layer + Downsampling)             
    '''
    # TODO: Why has he changed it to a pure convolutional instead avg pool?
    # TODO: No inital convolutional 7x7, stride 2 and Max Pool 3x3, stride 2.
    # TODO: It is possible these two are done to compensate Encoder/Decoder.
    def build_graph(self, x, training):
        self.training = training
        with tf.variable_scope('Encoder'):
            for i, layers in enumerate(self.blocks):

                # Dense Block.
                with tf.variable_scope('DenseBlock_%d' % i):
                    x = self.block(x, layers, self.growth_rates[i])

                # Squeeze-Excitation block.
                with tf.variable_scope('SE_DenseBlock_%d' % i):
                    x = squeeze_excitation_layer(x)

                # Transition Block.
                with tf.variable_scope('Transition_%d' % i):

                    # Last transition block: 2 output channels.
                    # Conv of 5x5 kernels, stride = 2.
                    filters = self.final_filters if i == len(self.blocks) - 1 else None
                    x = self.transition(x, filters=filters)

                # Squeeze-Excitation block.
                with tf.variable_scope('SE_Transition_%d' % i):
                    x = squeeze_excitation_layer(x)

        return tf.reshape(x, self.shape)


if __name__ == '__main__':
    print(Encoder().build_graph(tf.ones([25, 224, 224, 3]), True))
