import numpy as np
import readdata # self made library
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
import reportgen as rg

#Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def relu_back(x):
    return np.where(x>0,np.ones(x.shape),np.zeros(x.shape))


def sigmoid_back(x):
    return np.exp(x)/((1+np.exp(x))**2)


def leakyrelu(x):
    return np.where(x > 0, x, x * 0.01)
    #where function uses operation 1 for true condition, 2nd for false


def leakyrelu_back(x):
    return np.where(x > 0, 1, 0.01)


buffer = 10**-8
learning_rate = 0.002
epoch = 5
batch_size = 64
lamda = 0.05
report_name = "Report No 1"  # report name


def initialise_parameters(network):
    parameters = {} # empty dictionary
    np.random.seed(6)
    for i in range(1,len(network)):  # initialises weights and biases
        w_temp = np.random.randn(network[i-1], network[i])*np.sqrt(2/network[i-1])
        #weight initialization recommended by andrew ng in course 2
        b_temp = np.zeros((1, network[i]))
        parameters["W"+str(i)] = w_temp
        parameters["b"+str(i)] = b_temp
    return parameters


def forward_prop(A0, parameters):
    iter = len(parameters)//2  # no of weight arrays, i.e hidden layers+1
    A_temp = A0
    cache = {"A0": A0} # contains 2n+1 keys, n is no of layers
    for i in range(iter):
        Z_temp = np.dot(A_temp ,parameters["W"+str(i+1)]) + parameters["b"+str(i+1)]
        if i == iter - 1:  # last layer uses sigmoid
            A_temp = sigmoid(Z_temp)
        else:
            A_temp = leakyrelu(Z_temp)
        cache["Z" + str(i + 1)] = Z_temp
        cache["A" + str(i + 1)] = A_temp
    return A_temp, cache


def calc_cost(AL, Y_real, parameters=None):
    m = Y_real.shape[0]
    cost = (-1/m)*(np.sum(np.multiply(Y_real,np.log(AL+buffer))+np.multiply(1-Y_real,np.log(1-AL+buffer))))
    #cost function, buffer added to avoid log(0) error
    if lamda:
        for i in range(len(parameters)//2):
            cost += (-0.5/m)*lamda*np.sum(parameters["W"+str(i+1)]**2)
    cost = np.squeeze(cost)
    return cost


def backprop(AL,Y,caches, parameters):
    grads = {}
    L = len(caches)//2
    m = Y.shape[0]
    dAL = - (np.divide(Y, AL+buffer) - np.divide(1 - Y, 1 - AL + buffer))
    dZ = dAL*sigmoid_back(caches["Z"+str(L)])
    dW = (1 / m) * np.dot(caches["A"+str(L-1)].T, dZ) + (lamda/(2*m))*(parameters["W"+str(L)]**2)
    db = (1 / m) * np.sum(dZ, axis=0)
    dA_prev = np.dot(dZ, parameters["W" + str(L)].T)
    grads["dA" + str(L)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db.reshape(1, -1)
    for l in reversed(range(1,L)):
        dZ = dA_prev * leakyrelu_back(caches["Z" + str(l)])
        dW = (1 / m) * np.dot(caches["A" + str(l-1)].T, dZ) + (lamda/m)*parameters["W"+str(l)]
        db = (1 / m) * np.sum(dZ, axis=0)
        dA_prev = np.dot(dZ, parameters["W" + str(l)].T)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db.reshape(1, -1)
    return grads


def update_parameters(parameters, grads):
    L = len(parameters)//2
    for i in range(L):
        parameters["W" + str(i + 1)] -= learning_rate * grads["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] -= learning_rate * grads["db" + str(i + 1)]
    return #as parameters are passed by reference, no need to return


def train_model(X, Y_real, parameters):
    Y, cache = forward_prop(X, parameters)
    cost = calc_cost(Y, Y_real, parameters)
    costs_epoch = [cost]
    costs_batch = [cost]
    accuracies = [0]
    X, Y_real = shuffle(X, Y_real, random_state=batch_size)
    X, Y_real, X_dev, Y_dev = X[:1472], Y_real[:1472], X[1472:], Y_real[1472:]
    for i in range(epoch):
        X, Y_real = shuffle(X, Y_real, random_state=epoch)
        for j in range(math.ceil(X.shape[0]/batch_size)):
            X_batch = X[j * batch_size:((j + 1) * batch_size if ((j + 1) * batch_size) < X.shape[0] else X.shape[0] - 1)]
            Y_batch = Y_real[j * batch_size:((j + 1) * batch_size if ((j + 1) * batch_size) < X.shape[0] else X.shape[0] - 1)]
            Y, cache = forward_prop(X_batch, parameters)
            grads = backprop(Y, Y_batch, cache, parameters)
            update_parameters(parameters, grads)
            cost = calc_cost(Y, Y_batch, parameters)
            costs_batch.append(cost)
            print("Train Cost in Epoch number ",i+1,", Batch number ",j+1," is: ",cost)
        Y, cache = forward_prop(X, parameters)
        cost = calc_cost(Y, Y_real, parameters)
        accuracies.append(check_accuracy(X,Y_real,parameters))
        print("Train Set Accuracy after EPOCH ", i+1, " is: ", accuracies[i+1])
        print("Train Set Cost after EPOCH ", i+1, " is: ", cost)
        Y, cache = forward_prop(X_dev, parameters)
        print("Dev Set Cost after EPOCH ", i+1, " is: ",
              calc_cost(Y,Y_dev,parameters))
        print("Dev Set Accuracy after EPOCH ", i+1, " is: ",
              check_accuracy(X_dev,Y_dev,parameters))
        costs_epoch.append(cost)
    return costs_epoch, costs_batch, accuracies


def check_accuracy(X, Y_real, parameters):
    Y_ret, cache = forward_prop(X,parameters)
    m = Y_ret.shape[0]
    Y_real_max = np.squeeze(np.argmax(Y_real, axis=1))
    Y_ret_max = np.squeeze(np.argmax(Y_ret, axis=1))
    accuracy = 0
    for i in range(m):
        if Y_real_max[i] == Y_ret_max[i]:
            accuracy += 1
    return accuracy/m


def main():
    X_train, Y_train, X_test, Y_test = readdata.readshrink()
    X_train, X_test = X_train.reshape(X_train.shape[0],-1)/255, X_test.reshape(X_test.shape[0],-1)/255
    network = (X_train.shape[1], 1000, 400, 100, 40, Y_train.shape[1])
    parameters = initialise_parameters(network)
    costs_epoch, costs_batch, accuracies = train_model(X_train,Y_train,parameters)
    acc_train = check_accuracy(X_train,Y_train,parameters)
    acc_test = check_accuracy(X_test,Y_test, parameters)
    print("Train set Accuracy is: ", acc_train*100, "%")
    print("Test set Accuracy is: ", acc_test*100, "%")
    notes = input("Any Special Notes to add in the report: ")
    rg.imagegen(report_name , costs_epoch, costs_batch, accuracies)
    rg.reportgen(report_name, network, epoch,
                 learning_rate, acc_train, acc_test, costs_epoch[len(costs_epoch) - 1],
                 batch_size, lamda, notes)
    # this function call generates a report(HTML page) that details things
    # such as network structure, number of iterations, costs, accuracy, lambda
    # and various graphs.
    return



