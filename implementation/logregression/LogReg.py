import numpy as np


def sigmoid(s):
    return 1/(1+np.exp(-s))



def get_class_accuracy(X, y, theta):
    sample = np.dot(X, theta)
    pred = sigmoid(sample)
    count = 0
    for i in range(len(pred)):
        if pred[i] >= 0.5:
            count = count +1
    return count/len(y)



def getLossFunction(X, y, theta):
    losses = 0
    coly = y.shape[0]
    for i in range(len(y)):
        losses = losses + (-1)*np.log(sigmoid(np.dot(y[i], np.dot(X[i], theta))))
    loss = losses / coly
    return loss

def getGradient(X, y, theta):
    gradient = 0
    coly = y.shape[0]
    for i in range(len(y)):
        gradient = gradient + np.dot(np.dot(y[i], X[i]),(1-sigmoid(np.dot(y[i], np.dot(X[i], theta)))))
    grad = (-1)/coly * gradient
    return grad

def getStochGradient(X, y, theta,idx):
    gradient_idx = 0
    colIdx = len(idx)
    for i in idx:
        gradient_idx = gradient_idx + np.dot(np.dot(y[i], X[i]),(1-sigmoid(np.dot(y[i], np.dot(X[i], theta)))))
    grad_idx = (-1)/colIdx * gradient_idx
    return grad_idx
