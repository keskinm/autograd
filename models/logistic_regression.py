import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from lib.autograd import Graph, Tensor, Constant, Dot, Execution


class LogisticRegression:
    def __init__(self):
        pass

    def make_dataset(self):
        X, y = datasets.make_s_curve(100)
        W = np.random.normal(0, size=y.shape)
        return X, y, W

