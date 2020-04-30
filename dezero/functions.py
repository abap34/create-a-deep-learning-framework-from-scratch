import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils


class Sin(Function):
    def forward(self,x):
        y = np.sin(x)
        return y
    
    def backward(self,gy):
        x = self.inputs[0]
        return gy * cos(x)

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self,x):
        y = np.cos(x)
        return y
    
    def backward(self,gy):
        x = self.inputs[0]
        return gy * -sin(x)


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]() 
        gx = gy * (1 - y ** 2)
        return gx

def tanh(x):
    return Tanh()(x)

class MatMul(Function):
    def forward(self,x,w):
        y = x.dot(w)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy,W.T)     
        gW = matmul(x.T,gy) 
        return gx,gW

def matmul(x,W):
    return MatMul()(x,W)

class Reshape(Function):
    def __init__(self,shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self,gy):
        return reshape(gy,self.x_shape)

def reshape(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    def backward(self,gy):
        return transpose(gy)
    
def transpose(x):
    return Transpose()(x)

class BroadcastTo(Function):
    def __init__(self,shape):
        self.shape = shape
    def forward(self,x):
        self.x_shape = x.shape
        y = np.broadcast_to(x,self.shape)
        return y
    def backward(self,gy):
        gx = sum_to(gy,self.x_shape)
        return gx
        
def broadcast_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class Sum(Function):
    def __init__(self,axis,keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self,x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    def backward(self,gy):
        gy = utils.reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx = broadcast_to(gy,self.x_shape)
        return gx
        
def sum(x, axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
