from lib.autograd import Graph
from models.linear_regression import LinearRegression


def test_linear_regression():
    """Tests loss after train is lesser than 20% of original loss."""
    lr = LinearRegression()
    with Graph() as g:
        W, X, y = lr.init_tensors()
        lr.forward(W, X, g, y)
    loss = lr.loss
    lr.train_sample()
    assert lr.loss <= loss*0.20
