#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

torch.manual_seed(0)
plt.axis([0, 100, 0, 1])
plt.title('CPU training')
plt.ion()

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
x = Variable(torch.from_numpy(x_train), requires_grad=False).type(torch.FloatTensor)
y = Variable(torch.from_numpy(y_train), requires_grad=False).type(torch.FloatTensor)

# Test data
xt = Variable(torch.from_numpy(x_test), requires_grad=False).type(torch.FloatTensor)

# Model definition
model = torch.nn.Sequential(
    torch.nn.Linear(in_dim, hid_dim_1),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_dim_1, hid_dim_2),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_dim_2, hid_dim_3),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_dim_3, out_dim)
)

# Loss function
criterion = torch.nn.MSELoss()

# Optimization function
learning_rate = 1
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

for epoch in range(100):

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
    yt_pred = model(xt).data.numpy()
    accuracy = np.sum(np.argmax(y_test, axis=1) == np.argmax(yt_pred, axis=1)) / yt_pred.shape[0]

    print('Epoch: %d, Loss: %f, Accuracy: %f' % (epoch + 1, epoch_loss, accuracy))

    # Plot loss and accuracy
    plt.scatter(epoch, accuracy, c=['green'], s=4)
    plt.scatter(epoch, epoch_loss, c=['red'], s=4)
    plt.pause(0.05)

while True:
    plt.pause(0.05)