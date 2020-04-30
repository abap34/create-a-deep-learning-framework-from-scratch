if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable


x = Variable(np.random.randn(1,2,3))
y = x.reshape(6)
z = x.T

print(x)
print(y)
print(z)