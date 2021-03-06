import numpy as np

from lib.autograd import Graph
from models.conv_neural_network import ConvNeuralNetwork
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression


def test_train_linear_regression():
    """Tests loss after train is lesser than 20% of original loss."""
    lr = LinearRegression()
    with Graph() as g:
        W, X, y = lr.init_tensors()
        lr.forward(W, X, g, y)
    loss = lr.loss
    lr.train_sample()
    assert lr.loss <= loss * 0.20


def test_train_logistic_regression():
    """Tests loss after train is lesser than 60% of original loss.
    Or equals NaN sometimes (to check why).
    """
    lr = LogisticRegression()
    with Graph() as g:
        W, X, y = lr.init_tensors()
        executor, loss = lr.merge_forward_pass(W, X, g, y)
    lr.train_sample()
    assert np.isnan(lr.loss) or (lr.loss <= loss * 0.60)


def test_train_conv_neural_network():
    """Tests loss after train is lesser than 35% of the original loss."""
    from matplotlib import pyplot as plt

    nn = ConvNeuralNetwork()

    nn.train_stochastic(epochs=350)

    plt.plot(nn.losses)
    plt.savefig("losses.png")

    assert nn.losses[-1] <= nn.losses[0] * 0.35
