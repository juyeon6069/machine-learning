import numpy as np


def get_binary_prediction(soft_label):
    hard_label = [0 for i in range(len(soft_label))];
    for i in range(0, len(soft_label)):
        if soft_label[i] < 0:
            hard_label[i] = -1.0
        else:
            hard_label[i] = 1.0
    return hard_label
    
def get_misclassification_rate(y,yhat):
    incorrect_classified = 0
    for i in range(len(y)):
        if y[i] != yhat[i]:
            incorrect_classified += 1
    return incorrect_classified / len (y)
    

def get_accuracy_rate(y,yhat):
    correct_classified = 0
    for i in range(len(y)):
        if y[i] == yhat[i]:
            correct_classified += 1
    return correct_classified / len (y)

   
def get_avr_regression_error(y,yhat):
    re = (y - yhat) ** 2
    are = np.mean(re)
    return are
    
    
def get_RMSE(y,yhat):
    are = get_avr_regression_error(y,yhat)
    rmse = np.sqrt(are)
    return rmse
