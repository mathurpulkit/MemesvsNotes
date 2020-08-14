import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#used to suppress debug messages
import tensorflow as tf
import numpy as np
import readdata as rd
from numpy.random import seed
import random as pyrand
seed(1)
tf.random.set_seed(0)
np.random.seed(1)
pyrand.seed(7)
alpha = 0.001
epoch = 4
batchsize = 64


def mymodel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(32, 3, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2),
        tf.keras.layers.Conv2D(64, 5, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(input_shape=(32,32,64)),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.12),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dropout(0.12),
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dropout(0.12),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


def train_model():
    X_train, Y_train, X_test, Y_test = rd.read()
    X_train, X_test = X_train/255 - 0.5, X_test/255 - 0.5
    print(X_train.shape, Y_train.shape)
    model = mymodel()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fn, metrics=["accuracy"],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
    model.evaluate(X_test, Y_test)
    model.fit(X_train, Y_train, epochs=epoch,validation_split=0.0208,
              batch_size=batchsize, shuffle=True)
    model.evaluate(X_test, Y_test)


def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train_model()
