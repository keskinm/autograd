import uuid
import numpy as np

from lib.active_graph import active_graph

class Execution:
    def __init__(self, path):
        self.path = path

    def forward(self):
        for obj in reversed(self.path):
            if isinstance(obj, Operation):
                obj.value = obj.forward()

    def backward_ad(self):
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
        pass

class Graph:
    def __init__(self):
        self.phs = {}
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

    def compute_path(self, operation_id):
        operation = self.ops[operation_id]
        r = [operation]
        vis = set(operation.obj_id)

        def rec_compute_path(op):
            for inp in op.inputs:
                if not (inp.obj_id in vis):
                    r.append(inp)
                    vis.add(inp.obj_id)
                if isinstance(inp, Operation): rec_compute_path(inp)

        rec_compute_path(operation)
        return r


class InGraphObject:
    def __init__(self, name, obj_id=None):
        self.obj_id = obj_id or str(uuid.uuid4())
        self.name = name

    @staticmethod
    def check_other(other):
        if not isinstance(other, Tensor):
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
        return Mul(self, Constant(-1))

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
        return self.value

    @value.setter
    def value(self, value):
        raise ValueError("Cannot reassign constant")

class Operation(InGraphObject):
    def __init__(self, name=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].ops[self.obj_id] = self
        self.value = None
        self.grad = 0

    def forward(self):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    @property
    def inputs(self):
        raise NotImplementedError

class Add(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'add')
        self.a = a
        self.b = b

    @property
    def inputs(self):
        return [self.a, self.b]

    def forward(self):
        return self.a() + self.b()

    def backward(self, dout):
        return dout, dout

    def __call__(self, *args, **kwargs):
        return self.value

class Power(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'add')
        self.a = a
        self.b = b

    @property
    def inputs(self):
        return [self.a, self.b]

    def forward(self):
        return np.power(self.a(), self.b())

    def backward(self, dout):
        a = self.a()
        b = self.b()
        return dout*b*np.power(a, (b-1)), dout*np.log(a)*np.power(a, b)

    def __call__(self, *args, **kwargs):
        return self.value

class Divide(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'add')
        self.a = a
        self.b = b

    @property
    def inputs(self):
        return [self.a, self.b]

    def forward(self):
        return self.a() / self.b()

    def backward(self, dout):
        a = self.a()
        b = self.b()

        return dout/b, dout*a/np.power(b, 2)

    def __call__(self, *args, **kwargs):
        return self.value


class Mul(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'mul')
        self.a = a
        self.b = b

    @property
    def inputs(self):
        return [self.a, self.b]

    def forward(self):
        res =  self.a() * self.b()
        return res

    def backward(self, dout):
        return dout * self.b(), dout * self.a()

    def __call__(self, *args, **kwargs):
        return self.value


class MatMul(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'matmul')
        self.a = a
        self.b = b

    @property
    def inputs(self):
        return [self.a, self.b]

    def forward(self):
        res =  self.a() @ self.b()
        return res

    def backward(self, dout):
        return dout @ self.b().T, self.a().T @ dout

    def __call__(self, *args, **kwargs):
        return self.value

