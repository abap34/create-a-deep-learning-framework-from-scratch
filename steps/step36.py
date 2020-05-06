if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
print('gx:',gx)
x.cleargrad()
print('gx2:',gx)

z = gx ** 3 + y
print('z:',z)
z.backward()
print('gx3:',gx)
print('y:',y)
print(x.grad)
