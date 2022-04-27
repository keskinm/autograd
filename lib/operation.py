import numpy as np

from lib.abilities import WithGrad
from lib.active_graph import active_graph
from lib.autograd import InGraphObject


class Operation(InGraphObject, WithGrad):
    def __init__(self, name=None, compute_grads=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].ops[self.obj_id] = self
        self._value = None
        self.grad = None
        if compute_grads is not None:
            if not isinstance(compute_grads, list):
                raise TypeError
        self.compute_grads = compute_grads
        self.inputs = []

    @property
    def value(self):
        return self._value

    def find_input(self, find_id):
        for inp in self.inputs:
            if inp.id == find_id:
                return inp
        raise ValueError

    def get_to_compute_grads(self):
        if self.compute_grads is None:
            return self.inputs
        else:
            to_compute_grads = []
            for obj_id in self.compute_grads:
                to_compute_grads.append(self.find_input(obj_id))
        return to_compute_grads

    def compute_value(self):
        self._value = self.forward()

    def forward(self):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def add_inputs(self, inputs: list):
        if not isinstance(inputs, list):
            raise TypeError

        for inp in inputs:
            self.inputs.append(inp)

    def __call__(self, *args, **kwargs):
        return self._value


class Add(Operation):
    def __init__(self, a, b, name='add'):
        Operation.__init__(self, compute_grads=[a.id, b.id], name=name)
        self.a = a
        self.b = b
        self.add_inputs([self.a, self.b])

    def forward(self):
        return self.a() + self.b()

    def backward(self, dout):
        return dout, dout


class Power(Operation):
    def __init__(self, a, b, name='power'):
        Operation.__init__(self, compute_grads=[a.id, b.id], name=name)
        self.a = a
        self.b = b
        self.add_inputs([self.a, self.b])

    def forward(self):
        return np.power(self.a(), self.b())

    def backward(self, dout):
        a = self.a()
        b = self.b()
        l = dout*b*np.power(a, (b-1))
        r = dout*np.log(a)*np.power(a, b)
        return l, r


class Exp(Operation):
    def __init__(self, a, name='exp'):
        Operation.__init__(self, compute_grads=[a.id], name=name)
        self.a = a
        self.add_inputs([self.a])

    def forward(self):
        a = self.a()
        return np.exp(a)

    def backward(self, dout):
        return [dout * self.forward()]


class Log(Operation):
    def __init__(self, a, base=10, name='log'):
        Operation.__init__(self, compute_grads=[a.id], name=name)
        self.a = a
        self.base = base
        self.add_inputs([self.a])

    def forward(self):
        return getattr(np, f'log{self.base}')(self.a())

    def backward(self, dout):
        return [dout / self.a()]


class Divide(Operation):
    def __init__(self, a, b, name='divide'):
        Operation.__init__(self, compute_grads=[a.id, b.id], name=name)
        self.a = a
        self.b = b
        self.add_inputs([self.a, self.b])

    def forward(self):
        return self.a() / self.b()

    def backward(self, dout):
        a = self.a()
        b = self.b()

        return dout/b, dout*a/np.power(b, 2)


class Mul(Operation):
    def __init__(self, a, b, name='mul'):
        Operation.__init__(self, compute_grads=[a.id, b.id], name=name)
        self.a = a
        self.b = b
        self.add_inputs([self.a, self.b])

    def forward(self):
        a = self.a()
        b = self.b()
        res =  a * b
        return res

    def backward(self, dout):
        return dout * self.b(), dout * self.a()


class MatMul(Operation):
    def __init__(self, a, b, compute_grad=None, name='matmul'):
        compute_grad = compute_grad or [a.id, b.id]
        Operation.__init__(self, compute_grads=compute_grad, name=name)
        self.a = a
        self.b = b
        self.add_inputs([self.a, self.b])

    def forward(self):
        res =  self.a() @ self.b()
        return res

    def backward(self, dout):
        grads = []
        if self.a.id in self.compute_grads:
            a_grad = dout @ self.b().T
            grads.append(a_grad)

        if self.b.id in self.compute_grads:
            b_grad = self.a().T @ dout
            grads.append(b_grad)

        return grads


class Dot(Operation):
    def __init__(self, x, w, compute_grad=None, name='Dot'):
        compute_grad = compute_grad or [x.id, w.id]
        Operation.__init__(self, compute_grads=compute_grad, name=name)
        self.X = x
        self.W = w
        self.add_inputs([self.X, self.W])

    def forward(self):
        res =  np.dot(self.X(), self.W())
        return res

    def backward(self, dout):
        grads = []

        if self.X.id in self.compute_grads:
            x_grad = np.dot(dout, self.W().T)
            grads.append(x_grad)

        if self.W.id in self.compute_grads:
            w_grad = np.dot(self.X().T, dout)
            grads.append(w_grad)

        return grads


class Sum(Operation):
    def __init__(self, t, name='Sum'):
        Operation.__init__(self, compute_grads=[t.id], name=name)
        self.t = t
        self.add_inputs([self.t])

    def forward(self):
        return sum(self.t())

    def backward(self, dout):
        return dout * np.ones(self.t().shape)


class Conv2D(Operation):
    def __init__(self, x, w, compute_grad=None, name='conv2D'):
        compute_grad = compute_grad or [x.id, w.id]
        Operation.__init__(self, compute_grads=compute_grad, name=name)
        self.x = x
        self.w = w
        self.add_inputs([self.x, self.w])

    def forward(self):
        """
        self.w of shape (height, witdh) convolves self.X of shape (n_samples, height, width)
        :return: (n_samples, height, width) convolution
        """
        n_samples, x_dim_0, x_dim_1 = self.x().shape
        w_dim_0, w_dim_1 = self.w().shape

        samples_convolved = []

        for sample_idx in range(n_samples):

            convolved_to_reshape = []

            for left_corner in range(x_dim_0-w_dim_0+1):
                for top_corner in range(x_dim_1-w_dim_1+1):
                    mult = 0

                    for x in range(w_dim_0):
                        for y in range(w_dim_1):
                            mult += self.w()[x, y] * self.x()[sample_idx, left_corner+x, top_corner+y]

                    convolved_to_reshape.append(mult)

            convolved = np.array(convolved_to_reshape).reshape(x_dim_0-w_dim_0+1, x_dim_1-w_dim_1+1)

            samples_convolved.append(convolved)

        return np.stack(samples_convolved)

    def backward(self, dout):
        grads = []

        if self.x.id in self.compute_grads:
            raise NotImplementedError

        if self.w.id in self.compute_grads:
            n_samples, x_dim_0, x_dim_1 = self.x().shape
            w_dim_0, w_dim_1 = self.w().shape

            w_grad = []

            for x in range(w_dim_0):
                line_derivatives = []
                for y in range(w_dim_1):
                    contrib = 0

                    for sample_idx in range(n_samples):
                        for left_corner in range(x_dim_0 - w_dim_0 + 1):
                            for top_corner in range(x_dim_1 - w_dim_1 + 1):
                               contrib +=  dout[sample_idx, x, y] * self.x()[sample_idx, left_corner+x, top_corner+y]

                    line_derivatives.append(contrib)

                w_grad.append(line_derivatives)

            grads.append(np.array(w_grad))

        return grads


class Transpose(Operation):
    def __init__(self, tensor, transposition):
        Operation.__init__(self, name='transpose')
        self.t = tensor
        self.transposition = transposition
        self.add_inputs([self.t])

    def forward(self):
        return np.transpose(self.t(), self.transposition)

    def backward(self, dout):
        return np.transpose(dout, self.transposition)


class Flatten(Operation):
    def __init__(self, tensor, name='flatten'):
        Operation.__init__(self, name=name)
        self.t = tensor
        self.add_inputs([self.t])
        self.saved_shape = None

    def forward(self):
        arr = self.t()
        self.saved_shape = arr.shape
        return self.t().flatten()

    def backward(self, dout):
        return dout.reshape(self.saved_shape)
