import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from lib.autograd import Graph, Tensor, Constant, Dot, Execution, Sum


class LinearRegression:
    def __init__(self):
        pass

    def make_dataset(self):
        self.X, self.y = datasets.make_regression(n_samples=100, n_features=100)
        self.W = np.random.normal(0, size=self.y.shape)

    def make_graph(self):
        self.make_dataset()
        for _ in range(10):
            with Graph() as g:
                X = Tensor(self.X, name='X')
                y = Tensor(self.y, name='y')
                W = Tensor(self.W, name='W')
                z = Sum(Dot(X, W) + (-y**2))
                path = g.compute_path(z.obj_id)
                executor = Execution(path)
                executor.forward()
                print("z", z())
                executor.backward_ad()
                self.W = self.W - 0.001*W.grad

LinearRegression().make_graph()