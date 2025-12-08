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
    
    
class ReLU(Module):
    def __init__(self):
        pass
    
    def forward(self,s_in):
        return np.maximum(0, s_in)
    
    def backward(self,dL_ds_out,s_in):
        dL_ds_in = dL_ds_out * (s_in > 0)
        return dL_ds_in
        
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

    
class MNIST_2layer_binaryclass():
    def __init__(self, nhidden, istest=False):
        #self.layers = [Linear(100,784,0.01), ReLU(),   Linear(100,100,0.01),   ReLU(), Linear(10,100,0.01)]
        if istest:
            self.linear1 = Linear(nhidden,784, init_type = 'testcase')
        else:
            self.linear1 = Linear(nhidden,784, init_type = 'random')
        self.relu = ReLU()

        
        if istest:
            self.linear2 = Linear(1, nhidden, init_type = 'testcase')
        else:
            self.linear2 = Linear(1, nhidden, init_type = 'random')
            
        self.loss_fn = MSELoss()
        
        
    def forward(self,X,y):
        lin_result1 = self.linear1.forward(X)
        relu_result1 = self.relu.forward(lin_result1)
        lin_result2 = self.linear2.forward(relu_result1)
        self.loss = self.loss_fn.forward(y, lin_result2)
        yhat = lin_result2
        return np.mean(self.loss), yhat
        
    
    def backward(self,X,y):
        lin_result1 = self.linear1.forward(X)
        relu_result1 = self.relu.forward(lin_result1)
        lin_result2 = self.linear2.forward(relu_result1)
        
        dL_dyhat = self.loss_fn.backward(y, lin_result2)
        dL_drelu_result2 = self.linear2.backward(dL_dyhat, relu_result1) # self.linear2.update_var
        
        dL_dlin_result1 = self.relu.backward(dL_drelu_result2, lin_result1) #self.linear1.update_var
        dL_dX = self.linear1.backward(dL_dlin_result1, X)

        
        self.dL_dlin_result2 = dL_dyhat
        self.re1 = relu_result1
        
        self.dL_dlin_result1 = dL_dlin_result1
        self.s_relu_result1 =  X
            
    def update_params(self, stepsize):
        self.linear1.update_var(self.dL_dlin_result1, self.s_relu_result1, stepsize)
        self.linear2.update_var(self.dL_dlin_result2, self.re1, stepsize)
        
                
    
    def inference(self,X,y):
        loss, yhat = self.forward(X,y)
        yhat = np.sign(yhat)
        return loss, yhat           
