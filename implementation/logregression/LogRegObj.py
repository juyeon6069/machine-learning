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
    
    
class Loss():
    pass
    



    
class Sigmoid(Module):
    def __init__(self):
        pass
    
    def forward(self,s_in):
        return 1 / (1 + np.exp(-s_in))
    
    def backward(self,dL_ds_out,s_in):
        sig_out = self.forward(s_in)
        return dL_ds_out * sig_out * (1 - sig_out)
    
# f(y,hat y) = -y log(hat y) - (1-y) log(1-hat y)
class BCELoss(Loss):
    def __init__(self):
        pass
        
    def forward(self,y,yhat): 
        return -np.mean(y * np.log(np.maximum(yhat, 1e-2) ) + (1 - y) * np.log(np.maximum(1 - yhat, 1e-2) ))
    
    def backward(self,y,yhat):
        return -(y / (np.maximum(yhat, 1e-2) ) + (1 - y) / np.maximum(1 - yhat, 1e-2))
    
        

 
             
class MNIST_binaryclass():
    def __init__(self,istest =False):
        if istest:
            self.linear = Linear(1,784, init_type = 'testcase')
        else:
            self.linear = Linear(1,784, init_type = 'random')
        self.sigmoid = Sigmoid()
        self.loss_fn = BCELoss()

        
        self.states = [None,None,None]
        self.states_grad = [None,None,None]
        
    def forward(self,X,y):
        self.states[0] = X
        self.states[1] = self.linear.forward(self.states[0])
        yhat = self.states[1]
        
        self.states[2] = self.sigmoid.forward(self.states[1])
        loss = self.loss_fn.forward(y, yhat)
        return np.mean(loss), yhat
        
    
    def backward(self,X,y):
        self.states_grad[2] = self.loss_fn.backward(y, self.states[2]) 
        self.states_grad[1] = self.sigmoid.backward(self.states_grad[2], self.states[1])       
        self.states_grad[0] = self.linear.backward(self.states_grad[1], self.states[0])
            
            
    def update_params(self, stepsize):
        self.linear.update_var(self.states_grad[1], self.states[0], stepsize)
                
    #return hard prediction
    def inference(self,X,y):
        loss, yhat = self.forward(X,y)
        yhat = np.sign(yhat)
        return loss, yhat
       
