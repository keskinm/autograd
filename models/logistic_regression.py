import numpy as np
from matplotlib import pyplot as plt

from lib.autograd import Graph, Tensor, Constant, Dot, Execution, Divide, Exp, Log, Sum


class LogisticRegression:
    def __init__(self):
        pass

    def make_dataset(self, plot_dataset=False):
        self.X = np.array([[0.05*x, 0] for x in range(5)] + [[1-0.05*x, 1] for x in range(5)])
        self.y = np.array([0 for _ in range(5)] + [1 for _ in range(5)])
        if plot_dataset:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
            plt.savefig('plot_dataset.png')
        self.W = np.random.normal(0, size=self.X.shape[1])

    def train_sample(self):
        self.make_dataset()
        for _ in range(200):
            with Graph() as g:
                W, X, y = self.init_tensors()

                z1 = Divide(Constant(1), Exp(-Dot(X, W, relax_left=True) + Constant(1)))
                z2 = Divide(Constant(1), Exp(-Dot(X, W, relax_left=True) + Constant(1)))

                first_logged = Log(z1)
                loss_left = y*first_logged
                neg_loss_left = -loss_left
                second_logged = Log(-z2+Constant(1))
                loss_right = (-y+Constant(1, name='fourth const'))*second_logged
                loss_total = neg_loss_left + loss_right
                loss = Sum(loss_total)

                path, vis = g.compute_path(loss.obj_id)
                self.forward_backward_pass(W, loss, path)

    def init_tensors(self):
        X = Tensor(self.X, name='X')
        y = Tensor(self.y, name='y')
        W = Tensor(self.W, name='W')
        return W, X, y

    def train_sample_merging(self):
        self.make_dataset()
        for _ in range(200):
            with Graph() as g:
                W, X, y = self.init_tensors()

                dotted = Dot(X, W, relax_left=True)
                dotted.name = 'dotted'
                neg_dotted = -dotted
                neg_dotted.name = 'neg dotted'
                exped = Exp(neg_dotted + Constant(1, name='first const'))
                exped.name = 'exped'
                z = Divide(Constant(1, name='second const'), exped)
                z.name = 'z'
                first_logged = Log(z, name='first logged')
                loss_left = y*first_logged
                loss_left.name = 'loss_left'
                neg_loss_left = -loss_left
                neg_loss_left.name = 'neg_loss_left'
                second_logged = Log(-z+Constant(1, name='third const'), name='second logged')
                loss_right = (-y+Constant(1, name='fourth const'))*second_logged
                loss_right.name = 'loss_right'
                loss_total = neg_loss_left + loss_right
                loss_total.name = 'loss_total'
                loss = Sum(loss_total)
                loss.name = 'loss'

                z_path, z_vis = g.compute_path(z.obj_id)
                path, vis = g.compute_path(loss.obj_id, z_path, z_vis)
                self.forward_backward_pass(W, loss, path)

    def forward_backward_pass(self, W, loss, path):
        executor = Execution(path)
        executor.forward()
        print("loss", loss())
        executor.backward_ad()
        self.W = self.W - 0.001 * W.grad

