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
