if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F
import dezero.layers as L

np.random.seed(0)

x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1,l2]:
        for p in l.params():
            p.data -= lr * p.grad.data 
    
    if i % 1000 == 0:
        print(loss)


plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
test = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = predict(test)

plt.plot(test, y_pred.data, color='r')
plt.show()
