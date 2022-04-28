import random

import numpy as np
from sklearn import datasets

from lib.autograd import Graph, Constant, Execution, Tensor, PlaceHolder
from lib.operation import Dot, Sum, Flatten, BatchLessConv2D, Stack
from lib.optimizer import SGD


class ConvNeuralNetwork:
    def __init__(self):
        self.W = {}

        self.optimizer = SGD()
        self.losses = []

    def make_dataset_for_regression(self, n_samples=10, height=6, width=6):
        g = Graph()
        with g:
            X, xy_target = [], []
            for sample_idx in range(n_samples):
                s = np.zeros([height, width])
                _x, _y = random.randint(0, height - 1), random.randint(0, width - 1)
                s[_x, _y] = 1
                X.append(s)
                xy_target.append([_x, _y])

            X, xy_target = PlaceHolder(np.array(X)), PlaceHolder(np.array(xy_target))
            self.W["1"] = Tensor(np.random.normal(
                0, size=[X.value.shape[1] // 3, X.value.shape[2] // 3]
            ), name='W1')
            self.W["2"] = Tensor(np.random.normal(
                0, size=[X.value.shape[1] // 3, X.value.shape[2] // 3]
            ), name='W2')

            flattened_size = (X.value.shape[1] - X.value.shape[1] // 3) * (
                X.value.shape[2] - X.value.shape[2] // 3
            )

            self.W["x_pred"] = Tensor(np.random.normal(0, size=flattened_size), name='x_pred')
            self.W["y_pred"] = Tensor(np.random.normal(0, size=flattened_size), name='y_pred')
            self.W["xy_pred"] = Tensor(np.random.normal(0, size=[2, flattened_size]), name='xy_pred')
            self.optimizer.add_weights([self.W["1"],
                                        self.W["2"]])
        return g, X, xy_target

    def make_dataset_for_classification(self):
        datasets.load_digits()
        raise NotImplementedError

    def train_stochastic(self, epochs=200):
        graph, X, xy_target = self.make_dataset_for_regression()
        self.optimizer.add_weights([self.W["xy_pred"]])
        with graph:
            z1 = BatchLessConv2D(X, self.W["1"], compute_grad=[self.W["1"].id])
            z2 = BatchLessConv2D(z1, self.W["2"], compute_grad=[self.W["2"].id])
            z3 = Flatten(z2)
            xy_pred = Dot(self.W["xy_pred"], z3)
            loss = Sum((xy_pred + (-xy_target)) ** Constant(2))

        for epoch in range(epochs):
            epoch_loss = []
            for X_sample, xy_target_sample in list(zip(X, xy_target)):
                    path, vis = graph.compute_path(loss.obj_id)
                    executor = Execution(path)
                    executor.forward()
                    print(
                        "z1",
                        z1().shape,
                        "z2",
                        z2().shape,
                        "z3",
                        z3().shape,
                        "xy_pred",
                        xy_pred(),
                        "xy_real",
                        xy_target_sample(),
                        "loss",
                        loss(),
                    )
                    epoch_loss.append(loss())
                    executor.backward_ad()
                    self.optimizer.step()

            self.losses.append(sum(epoch_loss) / len(epoch_loss))

    def stochastic_duplicate_paths_draft(self, epochs=200):
        g, X, xy_target = self.make_dataset_for_regression()
        self.optimizer.add_weights([self.W["x_pred"], self.W["y_pred"],])
        with g:
            z11 = BatchLessConv2D(X, self.W["1"], compute_grad=[self.W["1"].id])
            z21 = BatchLessConv2D(z11, self.W["2"], compute_grad=[self.W["2"].id])
            z31 = Flatten(z21)

            z12 = BatchLessConv2D(X, self.W["1"], compute_grad=[self.W["1"].id])
            z22 = BatchLessConv2D(z12, self.W["2"], compute_grad=[self.W["2"].id])
            z32 = Flatten(z22)

            x_pred = Dot(z31, self.W["x_pred"], compute_grad=[self.W["x_pred"].id])
            y_pred = Dot(z32, self.W["y_pred"], compute_grad=[self.W["y_pred"].id])
            xy_pred = Stack([x_pred, y_pred])
            powered = (xy_pred + (-xy_target)) ** Constant(2)
            loss = Sum(powered, name='loss')

        for epoch in range(epochs):
            for X_sample, xy_target_sample in list(zip(X, xy_target)):
                path, vis = g.compute_path(loss.obj_id)
                executor = Execution(path)
                executor.forward()
                print("xy_pred", xy_pred(), "xy_real", xy_target_sample(), "loss", loss())
                executor.backward_ad()
                self.optimizer.step()
