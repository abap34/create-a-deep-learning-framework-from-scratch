import numpy as np
import weakref
import contextlib


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x



def as_tuple(x):
    if not isinstance(x,tuple):
        return (x,)
    return x



def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)

class Variable:
    __array_priotity__ = 200        # 演算子の優先度
    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError ('type:',type(data),'is not supported')
        self.data = data       # 変数の値の格納場所
        self.name = name
        self.grad = None
        self.creator = None    # 変数の生成元の関数を保持しておく
        self.generation = 0
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def set_creator(self,func):
        self.creator = func
        self.generation = func.generation + 1 # 優先度
    
    def clearglad(self):
        self.grad = None
    
    def backward(self,retaion_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            #print('grad is None. create',self.grad)
        funcs = []
        seen = set()
        def add_func(f):
            if f not in seen:
                funcs.append(f)
                seen.add(f)
                funcs.sort(key=lambda x : x.generation)
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            gxs = as_tuple(gxs)
            #print(f.name +".backward(",gys,") = ",gxs)
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:        # (x.cretor is None == x が入力変数) なのでそこでストップする
                    add_func(x.creator)
            if not retaion_grad:
                for y in f.outputs:
                    y().grad = None              # メモリを解放,yは弱参照なのでアクセスには()が必要

class Function:
    def __call__(self,*inputs:Variable) -> Variable:
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = as_tuple(self.forward(*xs))
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])    # 入力変数のうち最大の世代数を採用する
            for output in outputs:
                output.set_creator(self)
        self.inputs = inputs    # 入力変数を保持しておく
        self.outputs = [weakref.ref(output) for output in outputs] # 弱参照
        return outputs if len(outputs) > 1 else outputs[0]   
    
    def forward(self,xs):
        raise NotImplementedError()
    
    def backward(self,gys):
        raise NotImplementedError()

    
class Add(Function):
    name = "Add"
    def forward(self,x0,x1):
        return x0 + x1
    
    def backward(self,gy):
        return gy,gy

def add(x0,x1):
    x1 = as_array(x1)
    return Add()(x0,x1)



class Sub(Function):
    name = "Sub"
    def forward(self,x0,x1):
        return  x0 - x1
    
    def backward(self,gy):
        return gy, -gy
    
def sub(x0,x1):
    x1 = as_array(x1)
    return Sub()(x0,x1)


def rsub(x0,x1):           
    x1 = as_array(x1)
    return Sub()(x1,x0)



class Neg(Function):
    name = "Neg"
    def forward(self,x):
        return -x
    def backward(self,gy):
        return -gy

def neg(x):
    return Neg()(x)




class Mul(Function):
    name = "Mul"
    def forward(self,x0,x1):
        return x0 * x1
    
    def backward(self,gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
def mul(x0,x1):
    x1 = as_array(x1)
    return Mul()(x0,x1)



class Div(Function):
    name = "Div"
    def forward(self,x0,x1):
        return x0 / x1
    
    def backward(self,gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1  = gy * (-x0 / x1 ** 2)
        return gx0,gx1
    
    
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div(x1,x0)


class Config:
    enable_backprop = True


class Pow(Function):
    name = "Pow"
    def __init__(self,c):
        self.c = c
    
    def forward(self,x):
        return x ** self.c
    
    def backward(self,gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

        
def pow(x,c):
    return Pow(c)(x)



def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__neg__ = neg
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow



@contextlib.contextmanager
def using_config(name,value):
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)


def no_grad():
    return using_config('enable_backprop',False)