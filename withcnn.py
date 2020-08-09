import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#used to suppress debug messages
import tensorflow as tf
import numpy as np
import readdata as rd
from numpy.random import seed
import random as pyrand
seed(1)
tf.random.set_seed(2)
np.random.seed(3)
pyrand.seed(4)
alpha = 0.004
epoch = 8
batchsize = 32


def mymodel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(128, 5, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(4, 4),strides=4),
        #tf.keras.layers.Conv2D(128, 5, padding='same',activation='relu'),
        #tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        #tf.keras.layers.Dense(100, activation='relu'),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(80, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, activation=None)
    ])
    return model


def train_model():
    X_train, Y_train, X_test, Y_test = rd.read()
    #X_train = X_train.reshape((-1,256*256*3))/255
    #X_test = X_test.reshape((-1,256*256*3))/255
    X_train, X_test = X_train/255, X_test/255
    print(X_train.shape, Y_train.shape)
    model = mymodel()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fn,metrics=["accuracy"],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
    model.evaluate(X_test, Y_test)
    model.fit(X_train, Y_train, epochs=epoch,validation_split=0.0204,batch_size=batchsize)
    model.evaluate(X_test, Y_test)

def main():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train_model()
