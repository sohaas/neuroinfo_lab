import tensorflow as tf
import numpy as np

#TODO add comments, remove prints

"""
Model that does deconvolution on the meteorological data
to obtain a radar precipitation map
"""
class DeconvNet(tf.keras.Model):

    def __init__(self):
        super(DeconvNet, self).__init__()

        self.deconv_layers = [
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu)
        ]


    def call(self, x, training):

        #print("input: ", x.shape)

        x = np.expand_dims(x, axis=0)
        #print("expanded input: ", x.shape)

        for layer in self.deconv_layers:

            if (isinstance(layer, tf.keras.layers.BatchNormalization)):
              x = layer(x, training)
            else:
              x = layer(x)

        #print("output:", x.shape)

        x = tf.squeeze(x)
        #print("reshaped output:", x.shape)

        return x

