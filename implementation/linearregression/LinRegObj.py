import numpy as np


class Module():
    def __init__(self): 
        pass
    def forward(self,s_in):
        pass
    
    def backward(self, dL_ds_out, s_in):
        pass
    def update_var(self,dL_ds_out,s_in, stepsize):
        pass

    
    
class Linear(Module):
    def __init__(self, num_outparam,num_inparam, init_type = 'random'):
        if init_type == 'random':
            self.W = np.random.randn(num_outparam,num_inparam)*.01
            self.b = np.random.randn(num_outparam)*.01
        elif init_type == 'testcase':
            self.W = np.ones((num_outparam,num_inparam))
            self.b = np.ones((num_outparam))
        
    def forward(self,s_in):
        yhat = np.dot(self.W, s_in) + np.outer(self.b, np.ones(s_in.shape[1]))
        return yhat
    
    def backward(self,dL_ds_out,s_in):
        dL_ds_in = np.dot(self.W.T, dL_ds_out)
        return dL_ds_in
    
    def update_var(self,dL_ds_out,s_in, stepsize):
        s_inT = np.transpose(s_in)
        cols_in = s_in.shape[1]
        dW = np.dot(dL_ds_out, s_inT)
        db = np.sum(dL_ds_out, axis=1)
        
        self.W = self.W - dW*stepsize
        self.b = self.b - db*stepsize
        
        return 
    
    
## multiclass classification
class Loss():
    pass
    

class MSELoss(Loss):
    def __init__(self):
        pass
        
    def forward(self,y,yhat):
        e = y - yhat
        se = e ** 2
        return np.mean(se, axis=0) / 2
        
    def backward(self,y,yhat):
        e = y - yhat
        coly = y.shape[0]
        return (-1)*e/coly
    
    

class MNIST_binaryclass():
    def __init__(self, istest):
        if istest:
            self.linear = Linear(1,784, init_type = 'testcase')
        else:
            self.linear = Linear(1,784, init_type = 'random')
        self.loss_fn = MSELoss()

        
        self.states = [None,None]
        self.states_grad = [None,None]
        
    def forward(self,X,y):
        self.states[0] = X
        self.states[1] = self.linear.forward(self.states[0])
        loss = self.loss_fn.forward(self.states[1], y)
        yhat = self.states[1]
        return np.mean(loss), yhat
        
    
    def backward(self,X,y):
        yhat = self.states[1]
        self.states_grad[1] = self.loss_fn.backward(y, yhat)       
        self.states_grad[0] = self.linear.backward(self.states_grad[1], self.states[0])
            
            
    def update_params(self, stepsize):
        self.linear.update_var(self.states_grad[1], self.states[0], stepsize)
                
    #return hard prediction
    def inference(self,X,y):
        loss, yhat = self.forward(X,y)
        yhat = np.sign(yhat)
        return loss, yhat        
