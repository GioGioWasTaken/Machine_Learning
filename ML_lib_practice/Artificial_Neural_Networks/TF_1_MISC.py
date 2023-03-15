import tensorflow as tf
import numpy as np
import pandas as pd

# a model using tensorflow
model = tf.keras.models.Sequential( [
    tf.keras.layers.Dense( 25, activation="sigmoid" ),
    tf.keras.layers.Dense( 15, activation="sigmoid" ),
    tf.keras.layers.Dense( 1, activation="sigmoid" )
], name="my_model" )


### a model using numpy

def sigmoid(z):
    # sigmoid function, the activation we will be using.
    sigmoid = 1 / 1 + np.exp( -z )
    return sigmoid

def Dense(A_in, W, b, g):
    a_out = g( np.matmul( A_in, W ) + b )
    return a_out

def Sequential(X, W1, b1, W2, b2, W3, b3):
    a1 = Dense( X, W1, b1, sigmoid )
    a2 = Dense( a1, W2, b2, sigmoid )
    a3 = Dense( a2, W3, b3, sigmoid )
    return (a3)
