
if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt

import dezero
from dezero.datasets import Spiral
from dezero import DataLoader
from dezero.models import MLP
from dezero.functions import relu
from dezero.optimizers import SGD
from dezero.functions import softmax_cross_entropy_simple, accuracy

batch_size = 10
max_epoch = 300
hidden_size = 10
lr = 0.2

train = Spiral(train=True)
test = Spiral(train=False)

train_loder = DataLoader(train, batch_size)
test_loder = DataLoader(test, batch_size, shuffle=False)


model = MLP((hidden_size, 10),activation=relu)
optimizer = SGD(lr).setup(model)

train_acc = []
test_acc = []
train_loss = []
test_loss = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loder:
        y = model(x)
        loss = softmax_cross_entropy_simple(y, t)
        acc = accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    
    if epoch % 10 == 0: 
        print('epoch :', epoch + 1)
        print('train loss: {:.4f} accuracy : {:.4f}'.format(
            sum_loss / len(train), sum_acc / len(train)))
    train_acc.append(sum_acc)
    train_loss.append(sum_loss)
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loder:
            y = model(x)
            loss = softmax_cross_entropy_simple(y, t)
            acc = accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    if epoch % 10 == 0 : print('test loss: {:.4f} accuracy : {:.4f}'.format(sum_loss / len(train), sum_acc / len(train)))
    test_acc.append(sum_acc)
    test_loss.append(sum_loss)

plt.plot(range(max_epoch),train_acc)
plt.plot(range(max_epoch),test_acc)
plt.title("acc")
plt.xlabel("epoch")
plt.show()
