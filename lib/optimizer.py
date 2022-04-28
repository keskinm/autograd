class Optimizer:
    def __init__(self, weights=None):
        self.weights = weights or []

    def add_weights(self, weights):
        for weight in weights:
            self.weights.append(weight)

class SGD(Optimizer):
    def __init__(self, model_weights=None, lr=0.001):
        Optimizer.__init__(self, weights=model_weights)
        self.lr = lr

    def step(self):
        for weight in self.weights:
            weight.value -= self.lr * weight.grad
