from lib.autograd import Graph, Tensor, Execution

import numpy as np



def example1():
  val1, val2, val3 = np.array([0.9]), np.array([0.4]), np.array([1.3])
  with Graph() as g:
    x = Tensor(val1, name='x')
    y = Tensor(val2, name='y')
    c = Tensor(val3, name='c')

    z = (x*y+c)*c + x
    path = g.compute_path(z.obj_id)
    executor = Execution(path)
    executor.forward()
    assert z.value == (val1*val2+val3)*val3+val1
    executor.backward_ad()

    print([x.grad for x in executor.path])
    x_actual_expected = [g.tensors[x.obj_id].grad, val3*val2+1]
    y_actual_expected = [g.tensors[y.obj_id].grad, val1*val3]
    c_actual_expected = [g.tensors[c.obj_id].grad, val1*val2+2*val3]

    print(x_actual_expected)
    print(y_actual_expected)
    print(c_actual_expected)

example1()