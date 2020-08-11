#pillow needs to be installed to read jpg images


import matplotlib.image as img
from sklearn.utils import shuffle
import numpy as np
from PIL import  Image

def read():
    X = np.zeros((1600,256,256,3))
    Y = create_Y()
    for x in range(800):
        a = img.imread(r"Stage_2\Memes\\" + str(x) + ".jpg")
        X[x] = a
    for x in range(800):
        a = img.imread(r"Stage_2\Notes\\" + str(x) + ".jpg")
        X[x+800] = a
    X, Y = shuffle(X,Y,random_state=17)
    X_train, Y_train = X[:1536], Y[:1536]
    X_test, Y_test = X[1536:], Y[1536:]
    print(X.shape)
    return X_train, Y_train, X_test, Y_test


def readshrink():
    X = np.zeros((1600, 64, 64, 3))
    Y = create_Y()
    for x in range(800):
        a = Image.open(r"Stage_2\Memes\\" + str(x) + ".jpg")
        a = a.resize((64,64))
        X[x] = a
    for x in range(800):
        a = Image.open(r"Stage_2\Notes\\" + str(x) + ".jpg")
        a = a.resize((64,64))
        X[x+800] = a
    X, Y = shuffle(X,Y,random_state=17)
    X_train, Y_train = X[:1536], Y[:1536]
    X_test, Y_test = X[1536:], Y[1536:]
    print(X.shape)
    return X_train, Y_train, X_test, Y_test



def create_Y():
    Y = np.zeros((1600,2))
    for x in range(800):
        Y[x,0] = 1
    for x in range(800,1600):
        Y[x,1] = 1
    return Y