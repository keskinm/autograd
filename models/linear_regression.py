import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from lib.autograd import Graph, Tensor, Constant, Dot, Execution


class LinearRegression:
    def __init__(self):
        pass

    def make_dataset(self):
        X, y = datasets.make_regression()
        self.W = np.random.normal(0, size=y.shape)
        return X, y

    def make_graph(self):
        X, y = self.make_dataset()

        for _ in range(10):
            with Graph() as g:
                X = Tensor(X, name='X')
                y = Tensor(y, name='y')
                W = Tensor(self.W, name='W')
                z = Dot(X, W) + (-y**2)
                print("z", z)
                path = g.compute_path(z.obj_id)
                executor = Execution(path)
                executor.forward()
                executor.backward_ad()
                self.W = self.W - 0.001*W.grad

LinearRegression().make_graph()
