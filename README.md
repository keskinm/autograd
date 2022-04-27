## An implementation of Automatic differentiation library. 

Automatic differentiation is distinct from symbolic differentiation and numerical differentiation.


- Symbolic differentiation faces the difficulty of converting a computer program into a single 
mathematical expression and can lead to inefficient code. 
- Numerical differentiation (the method of finite differences) can introduce round-off errors in the discretization 
process and cancellation. 

Both of these classical methods have problems with calculating higher 
derivatives, where complexity and errors increase. Finally, both of these classical methods are 
slow at computing partial derivatives of a function with respect to many inputs, as is needed for 
gradient-based optimization algorithms. Automatic differentiation solves all of these problems.

It is this type of differentiation (AD) which is used in scientific computing frameworks as Torch and 
Tensorflow. Some example of gradient descent are provided within regular machine learning models. 

Here a snippet of usage: 

````python
from lib.autograd import Graph, Tensor, Execution, Constant
from lib.operation import Sum, Dot
def train_sample(self):
    for _ in range(200):
        with Graph() as g:
            X = Tensor(self.X, name='X')
            y = Tensor(self.y, name='y')
            W = Tensor(self.W, name='W')
    
            z = Sum((Dot(X, W) + (-y)) ** Constant(2))
            path, vis = g.compute_path(z.obj_id)
            executor = Execution(path)
            executor.forward()
            self.loss = z()
    
            print(f"Loss: {self.loss}")
            executor.backward_ad()
            self.W = self.W - 0.001*W.grad
````