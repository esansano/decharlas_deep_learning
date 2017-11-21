#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(0)

# Load data
train = pd.read_csv('train.csv', header=0).values
x = train[:, 1:] / 255.0
y = train[:, 0]
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1, 1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Neural network dimensions
in_dim = x_train.shape[1]
hid_dim_1 = 1000
hid_dim_2 = 500
hid_dim_3 = 100
out_dim = 10

# Batch size
batch = 1000

# Training data
x = Variable(torch.from_numpy(x_train), requires_grad=False).type(torch.FloatTensor).cuda()
y = Variable(torch.from_numpy(y_train), requires_grad=False).type(torch.FloatTensor).cuda()

# Test data
xt = Variable(torch.from_numpy(x_test), requires_grad=False).type(torch.FloatTensor).cuda()

# Model definition
model = torch.nn.Sequential(
    torch.nn.Linear(in_dim, hid_dim_1),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_dim_1, hid_dim_2),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_dim_2, hid_dim_3),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_dim_3, out_dim)
).cuda()

# Loss function
criterion = torch.nn.MSELoss().cuda()

# Optimization function
learning_rate = 1
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
loss_list = []
acc_list = []
max_epochs = 100

for epoch in range(max_epochs):

    epoch_loss = 0

    for i in range(0, x.size()[0], batch):
        # Forward pass
        y_pred = model(x[i:(i+batch), :])
        loss = criterion(y_pred, y[i:(i+batch), :])
        epoch_loss += loss.data[0]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

    # Test accuracy
    yt_pred = model(xt).data.cpu().numpy()
    accuracy = np.sum(np.argmax(y_test, axis=1) == np.argmax(yt_pred, axis=1)) / yt_pred.shape[0]

    print('Epoch: %d, Loss: %f, Accuracy: %f' % (epoch + 1, epoch_loss, accuracy))
    loss_list.append(epoch_loss)
    acc_list.append(accuracy)

# Plot loss and accuracy
plt.scatter(range(max_epochs), acc_list, c=['green'], s=4)
plt.scatter(range(max_epochs), loss_list, c=['red'], s=4)
plt.title('GPU training')
plt.axis([0, max_epochs, 0, 1])
plt.show()