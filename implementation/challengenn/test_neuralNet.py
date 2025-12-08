import numpy as np
import pickle
import sys
from neuralNet import *


 



def evaluate_trained_challenge( probnum, iseval):
    if iseval is False:
        model = pickle.load(open('models/challenge%s_trained.pkl' % probnum,'rb'))
        data = pickle.load(open('problems/challenge%s.pkl' % probnum,'rb'))
    else:
        data = pickle.load(open('eval_problems/challenge%s.pkl' % probnum,'rb'))
        
    
        model, runtime = master_train_model(probnum,data['Xtrain'],data['ytrain'])
        print('runtime: ', runtime)
        data = pickle.load(open('eval_problems/challenge%s.pkl' % probnum,'rb')) # challengee does not have access to this
    yhat = model.inference(data['Xtrain'])
    
    
    if probnum in ['1','2','3','4']:
        misclass = np.mean(np.not_equal(yhat, data['ytrain']))
        
        train_points = -np.log10(misclass)
    else:
        mse = np.mean((yhat-data['ytrain'])**2)
        train_points = max(0,1-mse)
        
    yhat = model.inference(data['Xtest'])
    if probnum in ['1','2','3','4']:
        misclass = np.mean(np.not_equal(yhat, data['ytest']))
        test_points = -np.log10(misclass)
    else:
        mse = np.mean((yhat-data['ytest'])**2)
        test_points = max(0,1-mse)
        
    return train_points , test_points




def master_train_model(probnum,Xtrain,ytrain):
    model = pickle.load(open('models/challenge%s_untrained.pkl' % probnum,'rb'))
    model = train(model, probnum, Xtrain, ytrain) # you write the train function
    return model
    
    
    
if __name__ == "__main__":
    probnum = sys.argv[1]
    iseval = sys.argv[2]
    if iseval == 'y':
        points = evaluate_trained_challenge( probnum, True)
    else:
        points = evaluate_trained_challenge( probnum, False)
    print(points)
  
