import random

import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from lib.autograd import Graph, Constant, Execution, Tensor
from lib.operation import Dot, Sum, Conv2D, Flatten


class ConvNeuralNetwork:
    def __init__(self):
        self.X = None
        self.y = None
        self.W1 = None
        self.W2 = None
        self.loss = None

    def make_dataset_for_regression(self, n_samples=10, height=6, width=6):
        X, y = [], []
        for sample_idx in range(n_samples):
            s = np.zeros([height, width])
            _x, _y = random.randint(0, height-1), random.randint(0, width-1)
            s[_x, _y] = 1
            X.append(s)
            y.append([_x, _y])

        self.X, self.y = np.array(X), y
        self.W1 = np.random.normal(0, size=[self.X.shape[1]//3, self.X.shape[2]//3])
        self.W2 = np.random.normal(0, size=[self.X.shape[1]//3, self.X.shape[2]//3])

    def make_dataset_for_classification(self):
        digits = datasets.load_digits()

    def draft(self, epochs=200):
        self.make_dataset_for_regression()

        for epoch in range(epochs):
            with Graph() as g:
                X = Tensor(self.X, name='X')
                y = Tensor(self.y, name='y')
                W1 = Tensor(self.W1, name='W1')
                W2 = Tensor(self.W2, name='W2')

                z1 = Conv2D(X, W1, compute_grad=[W1.id])
                z2 = Conv2D(z1, W2, compute_grad=[W2.id])
                # f = Flatten(z2)

                path, vis = g.compute_path(z2.obj_id)
                executor = Execution(path)
                executor.forward()
                print("z1", z1().shape, "z2", z2().shape)
                executor.backward_ad()


ConvNeuralNetwork().draft()
