import uuid

from lib.active_graph import active_graph

class Graph:
    def __init__(self):
        self.phs = {}
        self.tensors = {}
        self.ops = {}

    def __enter__(self):
        active_graph.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()

    def reset_session(self):
        for tensor in self.tensors.values():
            tensor.reset_grad()

class InGraphObject:
    def __init__(self, name, obj_id=None):
        self.obj_id = obj_id or str(uuid.uuid4())
        self.name = name

class PlaceHolder(InGraphObject):
    def __init__(self, value, name=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].phs[self.obj_id] = self
        self.value = value

class Tensor(InGraphObject):
    def __init__(self, value, name=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].tensors[self.obj_id] = self
        self.value = value
        self.grad = 0

    def reset_grad(self):
        self.grad = 0

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f'{other} must be a tensor')
        return Mul(self, other)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f'{other} must be a tensor')
        return Add(self, other)

class Operation(InGraphObject):
    def __init__(self, name=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].ops[self.obj_id] = self

class Add(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'add')
        self.a = a
        self.b = b

    def forward(self):
        return self.a + self.b

    def backward(self, dout):
        return self.b * dout, self.a * dout

class Mul(Operation):
    def __init__(self, a, b):
        Operation.__init__(self, 'mul')
        self.a = a
        self.b = b

    def forward(self):
        return self.a @ self.b

    def backward(self, dout):
        return dout @ self.b.T, self.a.T @ dout

