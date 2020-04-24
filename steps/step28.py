if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph


def rosenbrock(x0:Variable, x1:Variable):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y



x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 1e-3
iters = 10 ** 3 * 10

for i in range(iters):
    print(x0,x1)

    y = rosenbrock(x0,x1)

    x0.clearglad()  
    x1.clearglad()

    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

