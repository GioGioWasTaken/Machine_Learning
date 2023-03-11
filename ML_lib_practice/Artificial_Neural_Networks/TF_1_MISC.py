import tensorflow as tf
import numpy as np
import pandas as pd
x=np.array([[200,17]])
hidden1 = tf.keras.layers.Dense(inputs=x, units=25, activation=tf.nn.sigmoid)
hidden2 = tf.keras.layers.Dense(inputs=hidden1, units=15, activation=tf.nn.sigmoid)