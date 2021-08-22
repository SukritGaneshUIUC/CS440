# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()

        # Linear layers
        self.linear1 = torch.nn.Linear(in_size, 75)
        self.linear2 = torch.nn.Linear(75, out_size)

        # Loss function
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lrate)

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """

        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """

        self.train()
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return torch.mean(loss).item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """

    n_iter = 50

    # Transform Data
    N = len(train_set)
    for i in range(N):
        train_set[i] = (train_set[i] - torch.mean(train_set[i])) / torch.std(train_set[i])

    M = len(dev_set)
    for i in range(M):
        dev_set[i] = (dev_set[i] - torch.mean(dev_set[i])) / torch.std(dev_set[i])

    # Step 0: Create Model
    lrate = 0.0005
    criterion = torch.nn.CrossEntropyLoss()
    model = NeuralNet(lrate=lrate, loss_fn=criterion, in_size=32*32*3, out_size=2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lrate, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)

    # Step 1: Train
    losses = []

    for epoch in range(n_iter):
        print(epoch)
        running_loss = 0.0

        for i in range(0, N, batch_size):
            if (N - i > batch_size):
                currData = train_set[i:i+batch_size]
                currLabels = train_labels[i:i+batch_size]
            else:
                currData = train_set[i:]
                currLabels = train_labels[i:]

            running_loss += model.step(currData, currLabels)

        losses.append(running_loss)

    # Step 2: Test
    predictions = []

    for i in range(0, M, batch_size):
        if (M - i > batch_size):
            currData = dev_set[i:i+batch_size]
        else:
            currData = dev_set[i:]

        model.eval()
        output = model(currData)
        _, preds = torch.max(output, 1)
        predictions += preds.tolist()

    return losses, predictions, model
