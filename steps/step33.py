if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def f(x:Variable) -> Variable :
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i,x)

    y = f(x)
    x.clearglad()
    y.backward(create_graph=True)
    gx = x.grad
    x.clearglad()
    gx.backward()
    gx2 = x.grad
    x.data -= gx.data / gx2.data