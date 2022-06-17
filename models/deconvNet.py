import tensorflow as tf
import numpy as np

class DeconvNet(tf.keras.Model):
    """
    Model that does deconvolution on the meteorological data
    to obtain a radar precipitation map
    """

    def __init__(self):
        super(DeconvNet, self).__init__()

        self.deconv_layers = [
            # 61 x 72 x 72 x 143
            tf.keras.layers.Conv2DTranspose(143, kernel_size=(2, 2), strides=(2, 2), padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            # 61 x 72 x 72 x 72
            tf.keras.layers.Conv2DTranspose(72, kernel_size=(1, 1), strides=(1,1), padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            # 61 x 72 x 72 x 1
            tf.keras.layers.Conv2DTranspose(1, kernel_size=(1, 1), strides=(1,1), padding="valid"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.1),

            # 61 x 72 x 72 x 1
            tf.keras.layers.Dense(1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu)
        ]


    def call(self, x, training):

        # 61 x 36 x 36 x 143
        for layer in self.deconv_layers:

            if (isinstance(layer, tf.keras.layers.BatchNormalization)):
              x = layer(x, training)
            else:
              x = layer(x)

        # 61 x 72 x 72
        x = tf.squeeze(x)

        return x

