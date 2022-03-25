import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from lib.autograd import Graph, Tensor, Constant, Dot, Execution


class LinearRegression:
    def __init__(self):
        pass

    def make_dataset(self):
        X, y = datasets.make_regression()
        W = np.random.normal(0, size=y.shape)
        return X, y, W

    def make_graph(self):
        X, y, W = self.make_dataset()
        with Graph() as g:
            X = Constant(X, name='X')
            y = Constant(y, name='y')
            W = Tensor(W, name='W')
            z = Dot(X, W) + (-y**2)
            path = g.compute_path(z.obj_id)
            executor = Execution(path)
            executor.forward()
            # executor.backward_ad()

LinearRegression().make_graph()
