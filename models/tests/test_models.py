from lib.autograd import Graph
from models.conv_neural_network import ConvNeuralNetwork
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression


def test_linear_regression():
    """Tests loss after train is lesser than 20% of original loss."""
    lr = LinearRegression()
    with Graph() as g:
        W, X, y = lr.init_tensors()
        lr.forward(W, X, g, y)
    loss = lr.loss
    lr.train_sample()
    assert lr.loss <= loss*0.20

def test_logistic_regression():
    """Tests loss after train is lesser than 30% of original loss."""
    lr = LogisticRegression()
    with Graph() as g:
        W, X, y = lr.init_tensors()
        executor, loss = lr.merge_forward_pass(W, X, g, y)
    lr.train_sample()
    assert lr.loss <= loss*0.30

def test_conv_neural_network():
    """Tests loss after train is lesser than 40% of the original loss."""
    from matplotlib import pyplot as plt

    nn = ConvNeuralNetwork()
    nn.make_dataset_for_regression()

    nn.train_stochastic(epochs=200)

    plt.plot(nn.losses)
    plt.savefig('losses.png')

    assert nn.losses[-1] <= nn.losses[0]*0.10
