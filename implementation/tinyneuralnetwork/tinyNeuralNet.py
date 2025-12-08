import numpy as np


def sigmoid_fn(s):
    return 1/(1+np.exp(-s))


#############################


class Module():
    def __init__(self): 
        pass
    def forward(self,s_in):
        pass
    
    def backward(self, dL_ds_out, s_in):
        pass
    def update_var(self,dL_ds_out,s_in, stepsize):
        pass
    
    
class ReLU(Module):
    def __init__(self):
        pass
    
    def forward(self,s_in):
        return np.maximum(0, s_in)
    
    def backward(self,dL_ds_out,s_in):
        dL_ds_in = dL_ds_out * (s_in > 0)
        return dL_ds_in
    
    
class Sigmoid(Module):
    def __init__(self):
        pass
    
    def forward(self,s_in):
        return 1 / (1 + np.exp(-s_in))
    
    def backward(self,dL_ds_out,s_in):
        sig_out = self.forward(s_in)
        return dL_ds_out * sig_out * (1 - sig_out)
        
    
class Linear(Module):
    def __init__(self, num_outparam,num_inparam, init_type):
        if init_type == 1:
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
        
         
    
    
    
#####################################


class Model():
    def __init__(self):
        self.layers = [Linear(7, 6, 1), ReLU(), Linear(4, 7, 1)]
        self.loss_fn = CrossEntropyLoss()

    def forward(self, X, y):
        for layer in self.layers:
            X = layer.forward(X)
        loss = self.loss_fn.forward(X, y)
        return np.mean(loss), X

    def inference(self, X, y):
        predictions, loss = self.forward(X, y)
        yhat = np.argmax(predictions, axis=1)
        return loss, yhat

    
                
                
#################################################
## binary classification
class Loss():
    pass
    
    
class BCELoss(Loss):
    def __init__(self):
        pass
        
    def forward(self,y,yhat): 
        return -np.mean(y * np.log(np.maximum(yhat, 1e-2) ) + (1 - y) * np.log(np.maximum(1 - yhat, 1e-2) ))
    
    def backward(self,y,yhat):
        return -(y / (np.maximum(yhat, 1e-2) ) + (1 - y) / np.maximum(1 - yhat, 1e-2))
    
    

class SimpleReluClassNN(Model):
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # Layer 1
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = self.relu(self.z2)
        return output

    def relu_derivative(self, x):
        return x > 0

    def backward(self, x, y, yhat, learning_rate):
        d_z2 = (yhat - y) * self.relu_derivative(self.z2)
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.relu_derivative(self.z1)
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)

        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
    
############################################

## regression



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
    
    
class SimpleSigmoidRegressNN(Model):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.input = x
        self.output = self.sigmoid(np.dot(x, self.weights) + self.bias)
        return self.output

    def backward(self, y, yhat, learning_rate):
        error = y - yhat
        d_weights = np.dot(self.input.T, error * self.sigmoid_derivative(yhat))
        d_bias = np.sum(error * self.sigmoid_derivative(yhat), axis=0)

        self.weights += learning_rate * d_weights
        self.bias += learning_rate * d_bias

model = SimpleSigmoidRegressNN(input_size=3, output_size=1)

X = np.array([[0.1, 0.2, 0.3]])
y = np.array([[1]])

yhat = model.forward(X)

model.backward(y, yhat, learning_rate=0.01)
        
        
        
#############################################

class CrossEntropyLoss(Loss):
    def __init__(self):
        pass

    def forward(self, targets, predictions):
        self.softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        return -np.sum(targets * np.log(self.softmax + 1e-10), axis=1)

    def backward(self, targets, predictions):
        self.softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        return self.softmax - targets
    
    
class SimpleReluMulticlassNN(Model):
    def __init__(self, input_size, hidden_size, num_classes):
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
