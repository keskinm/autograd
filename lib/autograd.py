import uuid
import numpy as np

from lib.active_graph import active_graph

class Execution:
    def __init__(self, path):
        self.path = path

    def forward(self):
        """A forward pass of the path."""
        for obj in reversed(self.path):
            if isinstance(obj, Operation):
                obj.value = obj.forward()

    def backward_ad(self):
        """Backward automatic differentiation implementation."""
        vis = set()
        self.path[0].grad = 1
        for obj in self.path:
            if isinstance(obj, Tensor):
                continue
            grads = obj.backward(obj.grad)
            for inp, grad in zip(obj.inputs, grads):
                if isinstance(obj, Constant):
                    continue
                if not inp.obj_id in vis:
                    inp.grad = grad
                    vis.add(inp.obj_id)
                else:
                    inp.grad += grad

    def forward_ad(self):
        """Forward automatic differentiation implementation."""

class Graph:
    def __init__(self):
        self.tensors = {}
        self.ops = {}

    def __enter__(self):
        active_graph.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def reset_session(self):
        for tensor in self.tensors.values():
            tensor.reset_grad()

    def compute_path(self, operation_id, to_merge=None, to_merge_vis=None):
        n = []
        r = to_merge or []
        vis = to_merge_vis or set()

        operation = self.ops[operation_id]

        if operation not in r:
            n.append(operation)
        if operation.obj_id not in vis:
            vis.add(operation.obj_id)

        def rec_compute_path(op):
            for inp in op.inputs:
                if not (inp.obj_id in vis):
                    n.append(inp)
                    vis.add(inp.obj_id)
                if isinstance(inp, Operation): rec_compute_path(inp)

        rec_compute_path(operation)
        return n+r, vis


class InGraphObject:
    def __init__(self, name, obj_id=None):
        self.obj_id = obj_id or str(uuid.uuid4())
        self.name = name

    @staticmethod
    def check_other(other):
        if not isinstance(other, (Tensor, Operation, Constant)):
            if isinstance(other, float) or isinstance(other, int):
                other = Constant(value=other)
            else:
                other = Tensor(value=other)
        return other

    def __matmul__(self, other):
        other = self.check_other(other)
        return MatMul(self, other)

    def __mul__(self, other):
        other = self.check_other(other)
        return Mul(self, other)

    def __pow__(self, other):
        other = self.check_other(other)
        return Power(self, other)

    def __div__(self, other):
        other = self.check_other(other)
        return Divide(self, other)

    def __neg__(self):
        return Mul(self, Constant(value=-1))

    def __add__(self, other):
        other = self.check_other(other)
        return Add(self, other)

class Tensor(InGraphObject):
    def __init__(self, value, name=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].tensors[self.obj_id] = self
        self._value = value
        self.grad = 0

    def reset_grad(self):
        self.grad = 0

    def __call__(self, *args, **kwargs):
        return self._value

class Constant(Tensor):
    def __init__(self, value, name=None):
        super().__init__(value, name=name)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        raise ValueError("Cannot reassign constant")

class Operation(InGraphObject):
    def __init__(self, name=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].ops[self.obj_id] = self
        self.value = None
        self.grad = 0
        self.inputs = []

    def forward(self):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def add_inputs(self, inputs: list):
        for inp in inputs:
            self.inputs.append(inp)

    def __call__(self, *args, **kwargs):
        return self.value

class Add(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'add')
        self.a = a
        self.b = b
        self.add_inputs([self.a, self.b])

    def forward(self):
        return self.a() + self.b()

    def backward(self, dout):
        return dout, dout


class Power(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'power')
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
    def __init__(self, a):
        Operation.__init__(self, 'exp')
        self.a = a
        self.add_inputs([self.a])

    def forward(self):
        a = self.a()
        return np.exp(a)

    def backward(self, dout):
        return [dout * self.forward()]


class Log(Operation):
    def __init__(self, a, base=10, name='log'):
        Operation.__init__(self, name=name)
        self.a = a
        self.base = base
        self.add_inputs([self.a])

    def forward(self):
        return getattr(np, f'log{self.base}')(self.a())

    def backward(self, dout):
        return [dout / self.a()]


class Divide(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'divide')
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
    def __init__(self, a, b):
        Operation.__init__(self, 'mul')
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
    def __init__(self, a, b):
        Operation.__init__(self, 'matmul')
        self.a = a
        self.b = b
        self.add_inputs([self.a, self.b])

    def forward(self):
        res =  self.a() @ self.b()
        return res

    def backward(self, dout):
        return dout @ self.b().T, self.a().T @ dout


class Dot(Operation):
    def __init__(self, a, b, relax_left=None):
        Operation.__init__(self, 'dot')
        self.a = a
        self.b = b
        self.relax_left = relax_left
        self.add_inputs([self.a, self.b])

    def forward(self):
        res =  np.dot(self.a(), self.b())
        return res

    def backward(self, dout):
        a = self.a()
        b = self.b()

        left = self.relax_left or np.dot(dout, b.T)

        return left, np.dot(a.T, dout)

class Sum(Operation):
    def __init__(self, t):
        Operation.__init__(self, 'Sum')
        self.t = t
        self.add_inputs([self.t])

    def forward(self):
        return sum(self.t())

    def backward(self, dout):
        return dout * np.ones(self.t().shape)


class Conv2D(Operation):
    def __init__(self, x, w, name='conv2D'):
        Operation.__init__(self, name=name)
        self.x = x
        self.w = w
        self.add_inputs([self.x, self.w])

    def forward(self):
        x_dim_0, x_dim_1, n_samples = self.x().shape
        w_dim_0, w_dim_1 = self.w().shape

        samples_convolved = []

        for sample_idx in range(n_samples):

            convolved_to_reshape = []

            for left_corner in range(x_dim_0-w_dim_0+1):
                for top_corner in range(x_dim_1-w_dim_1+1):
                    mult = 0

                    for x in range(w_dim_0):
                        for y in range(w_dim_1):
                            mult += self.w()[x, y] * self.x()[left_corner+x, top_corner+y, sample_idx]

                    convolved_to_reshape.append(mult)

            convolved = np.array(convolved_to_reshape).reshape(x_dim_0-w_dim_0+1, x_dim_1-w_dim_1+1)

            samples_convolved.append(convolved)

        return samples_convolved

    def backward(self, dout):
        return

class Transpose(Operation):
    def __init__(self, arr, transposition):
        Operation.__init__(self, name='transpose')
        self.arr = arr
        self.transposition = transposition
        self.add_inputs([self.arr])

    def forward(self):
        return np.transpose(self.arr(), self.transposition)

    def backward(self, dout):
        # return np.transpose(dout, self.transposition)
        return
