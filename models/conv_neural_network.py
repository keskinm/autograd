import random

import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from lib.autograd import Graph, Constant, Dot, Execution, Sum, Conv2D
from models import Model


class ConvNeuralNetwork(Model):
    def __init__(self):
        super().__init__()

    def make_dataset_for_regression(self, n_samples=10, height=6, width=6):
        X, y = [], []
        for sample_idx in range(n_samples):
            s = np.zeros([height, width])
            _x, _y = random.randint(0, height-1), random.randint(0, width-1)
            s[_x, _y] = 1
            X.append(s)
            y.append([_x, _y])

        self.X, self.y = np.array(X).transpose([1, 2, 0]), y
        self.W = np.random.normal(0, size=[self.X.shape[0]//3, self.X.shape[1]//3])

    def make_dataset_for_classification(self):
        digits = datasets.load_digits()

    def draft(self):
        self.make_dataset_for_regression()

        for _ in range(200):
            with Graph() as g:
                W, X, y = self.init_tensors()

                z = Conv2D(X, W)
                path, vis = g.compute_path(z.obj_id)
                executor = Execution(path)
                executor.forward()
                print("z", z())

ConvNeuralNetwork().draft()
