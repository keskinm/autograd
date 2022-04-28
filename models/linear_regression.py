import numpy as np
from sklearn import datasets

from lib.autograd import Graph, Constant, Execution, Tensor
from lib.operation import Dot, Sum
from models import SimpleModel


class LinearRegression(SimpleModel):
    def __init__(self):
        super().__init__()

        self.make_dataset()
        self.init_weights(weights=None)

    def init_weights(self, weights=None):
        self.W = weights or np.random.normal(0, size=self.X.shape[1])

    def make_dataset(self):
        self.X, self.y = datasets.make_regression(n_samples=10, n_features=2)

    def train_sample(self, epochs=200):
        for epoch in range(epochs):
            with Graph() as g:
                W, X, y = self.init_tensors()
                executor = self.forward(W, X, g, y)
                print(f"Loss: {self.loss}")
                executor.backward_ad()
                self.W = self.W - 0.001*W.grad

    def stochastic_train_sample(self, epochs=200):
        for epoch in range(epochs):
            for X_sample, y_sample in list(zip(self.X, self.y)):
                with Graph() as g:
                    W = Tensor(self.W, name='W')
                    X = Tensor(X_sample, name='X_sample')
                    y = Tensor(y_sample, name='y_sample')
                    z = (Dot(X, W, compute_grad=[W.id]) + (-y)) ** Constant(2)
                    path, vis = g.compute_path(z.obj_id)
                    executor = Execution(path)
                    executor.forward()
                    self.loss = z()
                    print(f"Loss: {self.loss}")
                    executor.backward_ad()
                    self.W = self.W - 0.001 * W.grad

    def forward(self, W, X, g, y):
        z = Sum((Dot(X, W, compute_grad=[W.id]) + (-y)) ** Constant(2))
        path, vis = g.compute_path(z.obj_id)
        executor = Execution(path)
        executor.forward()
        self.loss = z()
        return executor
