import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#used to suppress debug messages
import tensorflow as tf
import numpy as np
import readdata as rd
from numpy.random import seed
import random as pyrand
from PIL import Image
seed(1)
tf.random.set_seed(2)
np.random.seed(3)
pyrand.seed(4)
alpha = 0.002
epoch = 10
batchsize = 64


def mymodel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


def train_model():
    X_train, Y_train, X_test, Y_test = rd.readshrink()
    X_train = X_train.reshape((-1, 64*64*3))/255 - 0.5
    X_test = X_test.reshape((-1, 64*64*3))/255 - 0.5
    #X_train, X_test = X_train/255 - 0.5, X_test/255 - 0.5
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
