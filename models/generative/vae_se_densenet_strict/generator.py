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

                # Introduce a Transition and SE blocks per dense layer, except for last one.
                if i != len(self.blocks) - 1:
                    # Transition Block.
                    with tf.variable_scope('Transition_%d' % i):
                        # Last transition block: final_filters=3
                        # Convolution transpose
                        filters = None if i < len(self.blocks) - 1 else self.final_filters
                        x = self.transition(x, filters=filters)

                    # Squeeze-Excitation block.
                    with tf.variable_scope('SE_Transition_%d' % i):
                        x = squeeze_excitation_layer(x)

            '''
            Last layers for mean and log square std.
            Transition --> Squeeze-Excitation for each.
            '''
            # Mean.
            with tf.variable_scope('Transition_mean_%d' % i):
                mean_xi_given_z = self.transition(x, filters=self.final_filters)
            with tf.variable_scope('SE_Transition_mean_%d' % i):
                mean_xi_given_z = squeeze_excitation_layer(mean_xi_given_z)
                mean_xi_given_z = tf.nn.sigmoid(mean_xi_given_z)

            # Log Square Std.
            with tf.variable_scope('Transition_logs2_%d' % i):
                logs2_xi_given_z = self.transition(x, filters=self.final_filters)
            with tf.variable_scope('SE_Transition_logs2_%d' % i):
                logs2_xi_given_z = squeeze_excitation_layer(logs2_xi_given_z)

        return mean_xi_given_z, logs2_xi_given_z


if __name__ == '__main__':
    print(Generator().build_graph(tf.ones([25, 56 * 56]), True))