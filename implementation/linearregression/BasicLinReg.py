import numpy as np

def solveLinearSystem(X,y):
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    XTX_inv = np.linalg.inv(XTX)
    XTy = np.dot(XT, y)
    theta = np.dot(XTX_inv, XTy)
    return theta


def getMeanSquaredError(y,ypred):
    se = (y-ypred)**2
    mse = np.mean(se)
    return mse

def predict(X,theta):
    predict_y = np.dot(X, theta) 
    return predict_y


def get_grad(X,y,theta):
    XT = np.transpose(X)
    pred = np.dot(X, theta)
    colX = X.shape[0]
    pred_min_y = pred - y
    XT_pred_min_y = np.dot(XT, pred_min_y)
    grad = 1/colX * XT_pred_min_y
    return grad


def run_gradient_descent(X,y,theta0, stepsize, max_steps):
    theta = theta0
    for k in range(max_steps):
        theta = theta - stepsize*get_grad(X,y,theta)
    return theta
