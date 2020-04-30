if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import  dezero.functions as F
from dezero import Variable

def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)


x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0)
print
y.backward()
print(y)
print(x.grad)
print('=======')
x = Variable(np.random.randn(2,3,4,5))
y =x.sum(keepdims=True)
print(y.shape)