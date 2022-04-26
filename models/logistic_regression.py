import numpy as np
from matplotlib import pyplot as plt

from lib.autograd import Graph, Constant, Dot, Execution, Divide, Exp, Log, Sum
from models.models import Model


class LogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self.make_dataset()
        self.init_weights()

    def make_dataset(self, plot_dataset=False):
        self.X = np.array([[0.05*x, 0] for x in range(5)] + [[1-0.05*x, 1] for x in range(5)])
        self.y = np.array([0 for _ in range(5)] + [1 for _ in range(5)])
        if plot_dataset:
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y)
            plt.savefig('plot_dataset.png')

    def train_sample(self, forward_style='merge_forward_pass'):
        self.make_dataset()
        for _ in range(200):
            with Graph() as g:
                W, X, y = self.init_tensors()

                executor, loss = getattr(self, forward_style)(W, X, g, y)
                self.loss = loss
                print(f"Loss: {self.loss}")
                executor.backward_ad()
                self.W += 0.001 * W.grad

    def forward_pass(self, W, X, g, y):
        z1 = Divide(Constant(1), Exp(-Dot(X, W, relax_left=True) + Constant(1)))
        z2 = Divide(Constant(1), Exp(-Dot(X, W, relax_left=True) + Constant(1)))
        first_logged = Log(z1)
        loss_left = y * first_logged
        neg_loss_left = -loss_left
        second_logged = Log(-z2 + Constant(1))
        loss_right = (-y + Constant(1, name='fourth const')) * second_logged
        loss_total = neg_loss_left + loss_right
        loss = Sum(loss_total)
        path, vis = g.compute_path(loss.obj_id)
        executor = Execution(path)
        executor.forward()
        return executor, loss()

    def merge_forward_pass(self, W, X, g, y):
        dotted = Dot(X, W, relax_left=True)
        dotted.name = 'dotted'
        neg_dotted = -dotted
        neg_dotted.name = 'neg dotted'
        exped = Exp(neg_dotted + Constant(1, name='first const'))
        exped.name = 'exped'
        z = Divide(Constant(1, name='second const'), exped)
        z.name = 'z'
        first_logged = Log(z, name='first logged')
        loss_left = y * first_logged
        loss_left.name = 'loss_left'
        neg_loss_left = -loss_left
        neg_loss_left.name = 'neg_loss_left'
        second_logged = Log(-z + Constant(1, name='third const'), name='second logged')
        loss_right = (-y + Constant(1, name='fourth const')) * second_logged
        loss_right.name = 'loss_right'
        loss_total = neg_loss_left + loss_right
        loss_total.name = 'loss_total'
        loss = Sum(loss_total)
        loss.name = 'loss'
        z_path, z_vis = g.compute_path(z.obj_id)
        path, vis = g.compute_path(loss.obj_id, z_path, z_vis)
        executor = Execution(path)
        executor.forward()
        return executor, loss()
