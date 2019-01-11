import tensorflow as tf
from models.generative.gan.encoder import Encoder


class Discriminator(Encoder):
    def __init__(self):
        super().__init__()
        self.classes = 1
        self.shape = [-1, 56, 56, 2]

    def build_graph(self, x, training, reuse=None):
        self.training = training
        with tf.variable_scope('Discriminator', reuse=reuse):
            x = super().build_graph(x, training)
            x = tf.reshape(x, self.shape)
            with tf.variable_scope('Classification'):
                x = self.classification(x)
            x = tf.nn.sigmoid(x)
        return x


if __name__ == '__main__':
    print(Discriminator().build_graph(tf.ones([25, 224, 224, 3]), True))
