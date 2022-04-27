from lib.autograd import Graph, Tensor, Execution, Constant, Conv2D


def test_example1():
    """Check autodiff derivatives equals symbolic ones."""
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
    import numpy as np
    import tensorflow as tf

    X = np.random.normal(0, size=[2, 2, 10])

    X_tf_input = X.transpose([2, 0, 1])
    X_tf_input = np.expand_dims(X_tf_input, axis=-1)

    print("X_tf_input shape", X_tf_input.shape)
    print("given shape", X_tf_input.shape[1:])

    tf_conv2D = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=3,
        strides=(1, 1),
        padding=('valid'),
        data_format='channels_last',
        activation=None,
        dilation_rate=2,
        input_shape=X_tf_input.shape[1:]
    )
    weights = tf_conv2D.get_weights()
    tf_forwarded = tf_conv2D(X_tf_input)

    with Graph() as g:
        z = Conv2D(X, weights)
        path, vis = g.compute_path(z.obj_id)
        executor = Execution(path)
        executor.forward()

    assert z() == tf_forwarded


test_conv2D()