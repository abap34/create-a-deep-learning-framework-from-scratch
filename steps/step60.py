if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

import dezero
from dezero.utils import plot_dot_graph
from dezero import Model
from dezero.dataloders import SeqDataLoader
import dezero.functions as F
import dezero.layers as L
from dezero.optimizers import Adam

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30


train = dezero.datasets.SinCurve(train=True)
dataloder = SeqDataLoader(train, batch_size=batch_size)
seqlen = len(train)

class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y
    
model = BetterRNN(hidden_size, 1)
optimizer = Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0
    for x, t in dataloder:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1
        if epoch == 0 and count == 5:
            plot_dot_graph(loss, to_file='RNNwithLSTM.png')
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        avg_loss = float(loss.data) / count

    print('epoch', epoch + 1, 'loss',avg_loss)

plot_dot_graph(loss)
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
