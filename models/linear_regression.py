import numpy as np
from sklearn import datasets

from lib.autograd import Graph, Tensor, Constant, Dot, Execution, Sum


class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        self.W = None
        self.loss = None

        self.make_dataset()
        self.init_weights(weights=None)

    def init_weights(self, weights=None):
        self.W = weights or np.random.normal(0, size=self.X.shape[1])

    def make_dataset(self):
        self.X, self.y = datasets.make_regression(n_samples=10, n_features=2)

    def train_sample(self):
        for _ in range(200):
            with Graph() as g:
                W, X, y = self.init_tensors()
                executor = self.forward(W, X, g, y)
                print(f"Loss: {self.loss}")
                executor.backward_ad()
                self.W = self.W - 0.001*W.grad

    def init_tensors(self):
        X = Tensor(self.X, name='X')
        y = Tensor(self.y, name='y')
        W = Tensor(self.W, name='W')
        return W, X, y

    def forward(self, W, X, g, y):
        z = Sum((Dot(X, W, relax_left=True) + (-y)) ** Constant(2))
        path, vis = g.compute_path(z.obj_id)
        executor = Execution(path)
        executor.forward()
        self.loss = z()
        return executor
