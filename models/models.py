import numpy as np

from lib.autograd import Tensor


class SimpleModel:
    def __init__(self):
        self.X = None
        self.y = None
        self.W = None
        self.loss = None

    def init_weights(self, weights=None):
        self.W = weights or np.random.normal(0, size=self.X.shape[1])

    def init_tensors(self):
        X = Tensor(self.X, name="X")
        y = Tensor(self.y, name="y")
        W = Tensor(self.W, name="W")
        return W, X, y
