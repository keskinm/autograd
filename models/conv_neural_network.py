import random

import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from lib.autograd import Graph, Constant, Execution, Tensor
from lib.operation import Dot, Sum, Conv2D, Flatten, Reshape, BatchLessConv2D, Stack


class ConvNeuralNetwork:
    def __init__(self):
        self.X = None
        self.y = None
        self.W = None
        self.loss = None
        self.losses = []

    def make_dataset_for_regression(self, n_samples=10, height=6, width=6):
        X, y = [], []
        for sample_idx in range(n_samples):
            s = np.zeros([height, width])
            _x, _y = random.randint(0, height-1), random.randint(0, width-1)
            s[_x, _y] = 1
            X.append(s)
            y.append([_x, _y])

        self.X, self.y = np.array(X), np.array(y)
        self.W1 = np.random.normal(0, size=[self.X.shape[1]//3, self.X.shape[2]//3])
        self.W2 = np.random.normal(0, size=[self.X.shape[1]//3, self.X.shape[2]//3])

        flattened_size = (self.X.shape[1] - self.X.shape[1]//3) * (self.X.shape[2] - self.X.shape[2]//3)

        self.W_x_pred = np.random.normal(0, size=flattened_size)
        self.W_y_pred = np.random.normal(0, size=flattened_size)
        self.W_xy_pred = np.random.normal(0, size=[2, flattened_size])

    def make_dataset_for_classification(self):
        digits = datasets.load_digits()

    def train_stochastic(self, epochs=200):
        self.make_dataset_for_regression()

        for epoch in range(epochs):
            for X_sample, y_sample in list(zip(self.X, self.y)):
                with Graph() as g:
                    W1, W2, W_xy_pred, executor = self.forward_stochastic(X_sample, g, y_sample)
                    executor.backward_ad()
                    self.W1 = self.W1 - 0.001 * W1.grad
                    self.W2 = self.W2 - 0.001 * W2.grad
                    self.W_xy_pred = self.W_xy_pred - 0.001 * W_xy_pred.grad

    def forward_stochastic(self, X_sample, g, y_sample):
        X = Tensor(X_sample, name='X')
        y = Tensor(y_sample, name='y')
        W1 = Tensor(self.W1, name='W1')
        W2 = Tensor(self.W2, name='W2')
        W_xy_pred = Tensor(self.W_xy_pred, name='W_xy_pred')
        z1 = BatchLessConv2D(X, W1, compute_grad=[W1.id])
        z2 = BatchLessConv2D(z1, W2, compute_grad=[W2.id])
        z3 = Flatten(z2)
        xy_pred = Dot(W_xy_pred, z3)
        loss = Sum((xy_pred + (-y)) ** Constant(2))
        path, vis = g.compute_path(loss.obj_id)
        executor = Execution(path)
        executor.forward()
        print("z1", z1().shape, "z2", z2().shape, "z3", z3().shape,
              "xy_pred", xy_pred(), "xy_real", y(), "loss", loss())
        self.loss = loss()
        self.losses.append(self.loss)
        return W1, W2, W_xy_pred, executor

    def stochastic_duplicate_paths_draft(self, epochs=200):
        self.make_dataset_for_regression()

        for epoch in range(epochs):
            for X_sample, y_sample in list(zip(self.X, self.y)):
                with Graph() as g:
                    X = Tensor(X_sample, name='X')
                    y = Tensor(y_sample, name='y')
                    W1 = Tensor(self.W1, name='W1')
                    W2 = Tensor(self.W2, name='W2')

                    z11 = BatchLessConv2D(X, W1, compute_grad=[W1.id])
                    z21 = BatchLessConv2D(z11, W2, compute_grad=[W2.id])
                    z31 = Flatten(z21)

                    z12 = BatchLessConv2D(X, W1, compute_grad=[W1.id])
                    z22 = BatchLessConv2D(z12, W2, compute_grad=[W2.id])
                    z32 = Flatten(z22)

                    W_x_pred = Tensor(self.W_x_pred, name='W_x_pred')
                    W_y_pred = Tensor(self.W_y_pred, name='W_y_pred')
                    x_pred = Dot(z31, W_x_pred, compute_grad=[W_x_pred.id])
                    y_pred = Dot(z32, W_y_pred, compute_grad=[W_y_pred.id])
                    xy_pred = Stack([x_pred, y_pred])
                    powered = (xy_pred + (-y)) ** Constant(2)
                    loss = Sum(powered)

                    path, vis = g.compute_path(loss.obj_id)
                    executor = Execution(path)
                    executor.forward()
                    print("xy_pred", xy_pred(), "xy_real", y(), "loss", loss())
                    executor.backward_ad()
                    self.W1 = self.W1 - 0.001 * W1.grad
                    self.W2 = self.W2 - 0.001 * W2.grad
                    self.W_x_pred = self.W_x_pred - 0.001 * W_x_pred.grad
                    self.W_y_pred = self.W_y_pred - 0.001 * W_y_pred.grad
