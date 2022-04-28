import uuid


from lib.abilities import WithGrad
from lib.active_graph import active_graph

import sys

if sys.version_info[0:3] != (3, 6, 9):
    raise Exception("Requires python 3.6.9")


class Execution:
    def __init__(self, path):
        self.path = path

    def forward(self):
        """A forward pass of the path."""
        for obj in reversed(self.path):
            obj.compute_value()

    def backward_ad(self):
        """Backward automatic differentiation implementation."""
        vis = set()
        self.path[0].init_grad("ones_like")
        for obj in self.path[1:]:
            if isinstance(obj, WithGrad):
                obj.init_grad("zeros_like")

        for obj in self.path:
            if isinstance(obj, (Tensor, PlaceHolder)):
                continue

            grads = obj.backward(obj.grad)
            for inp, grad in zip(obj.get_to_compute_grads(), grads):
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
        self.placeholders = {}

    def __enter__(self):
        active_graph.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def reset_session(self):
        for tensor in self.tensors.values():
            tensor.reset_grad()

    def compute_path(self, operation_id, to_merge=None, to_merge_vis=None):
        from lib.operation import Operation

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
                if isinstance(inp, Operation):
                    rec_compute_path(inp)

        rec_compute_path(operation)
        return n + r, vis


class InGraphObject:
    def __init__(self, name, obj_id=None):
        self.obj_id = obj_id or str(uuid.uuid4())
        self.name = name

    @property
    def id(self):
        return self.obj_id

    def compute_value(self):
        pass

    @staticmethod
    def check_other(other):
        if not isinstance(other, InGraphObject):
            if isinstance(other, float) or isinstance(other, int):
                other = Constant(value=other)
            else:
                other = Tensor(value=other)
        return other

    def __matmul__(self, other):
        from lib.operation import MatMul

        other = self.check_other(other)
        return MatMul(self, other)

    def __mul__(self, other):
        from lib.operation import Mul

        other = self.check_other(other)
        return Mul(self, other)

    def __pow__(self, other):
        from lib.operation import Power

        other = self.check_other(other)
        return Power(self, other)

    def __div__(self, other):
        from lib.operation import Divide

        other = self.check_other(other)
        return Divide(self, other)

    def __neg__(self):
        from lib.operation import Mul

        return Mul(self, Constant(value=-1))

    def __add__(self, other):
        from lib.operation import Add

        other = self.check_other(other)
        return Add(self, other)


class Tensor(InGraphObject, WithGrad):
    def __init__(self, value, name=None):
        WithGrad.__init__(self)
        InGraphObject.__init__(self, name=name)
        active_graph[-1].tensors[self.obj_id] = self
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

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


class PlaceHolder(InGraphObject):
    def __init__(self, value, name=None):
        InGraphObject.__init__(self, name=name)
        active_graph[-1].placeholders[self.obj_id] = self
        self._value = value
        self.idx = 0

    @property
    def value(self):
        return self._value

    def __call__(self, *args, **kwargs):
        return self._value[self.idx]

    def __next__(self):
        if self.idx == len(self._value) - 1:
            raise StopIteration
        self.idx += 1
        return self

    def __iter__(self):
        self.idx = 0
        return self
