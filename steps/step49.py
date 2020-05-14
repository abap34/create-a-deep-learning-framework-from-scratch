
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from math import *


import dezero
from dezero import optimizers

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train = dezero.datasets.Spiral()

model = dezero.models.MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train)
max_iter = ceil(data_size / batch_size)

loss_result = []
for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0
    for i in range(max_iter):
        batch_idx = index[i * batch_size:(i + 1) * batch_size]
        batch = [train[i] for i in batch_idx]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = dezero.functions.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    loss_result.append(avg_loss)
    print('epoch %d, loss %.3f' % (epoch + 1, avg_loss))


plt.plot(loss_result)
plt.show()