import numpy as np

class WithValue:
    def __init__(self, value):
        """Object which holds value."""
        self._value = value

class WithGrad(WithValue):
    def __init__(self, value=None):
        """Object which holds gradient."""
        WithValue.__init__(self, value=value)
        self.grad = None

    def init_grad(self, np_init):
        maps = {
            'zeros_like': np.zeros_like,
            'ones_like': np.ones_like
        }
        self.grad = maps[np_init](self._value)

    def reset_grad(self):
        self.grad = np.zeros_like(self._value)
