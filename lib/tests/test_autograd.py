from lib.autograd import Graph, Tensor, Execution, Constant, Conv2D


def test_example1():
    val1, val2, val3 = 0.9, 0.4, 1.3
    with Graph() as g:
        x = Tensor(val1, name='x')
        y = Tensor(val2, name='y')
        c = Constant(val3, name='c')

        z = (x*y+c)*c + x
        path, vis = g.compute_path(z.obj_id)
        executor = Execution(path)
        executor.forward()
        assert z.value == (val1*val2+val3)*val3+val1
        executor.backward_ad()

        assert g.tensors[x.obj_id].grad == val3*val2+1
        assert g.tensors[y.obj_id].grad == val1*val3
        assert g.tensors[c.obj_id].grad == val1*val2+2*val3


def test_conv2D():
    """Tests Conv2D outputs same results as torch Conv2D."""
    # from keras import Conv2D as kConv2D
