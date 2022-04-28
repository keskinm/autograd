## An implementation of Automatic differentiation. 

Classical methods for computing derivatives includes: 

- Symbolic differentiation faces the difficulty of converting a computer program into a single 
mathematical expression and can lead to inefficient code. 
- Numerical differentiation (the method of finite differences) can introduce round-off errors in the discretization 
process and cancellation. 

Both of these classical methods have problems with calculating higher 
derivatives, where complexity and errors increase. These classical methods are 
slow at computing partial derivatives of a function with respect to many inputs, as is needed for 
gradient-based optimization algorithms. 


Automatic differentiation is an efficient method to compute derivatives that solves these problems.

````python
from lib.autograd import Graph, Tensor, Execution, Constant
from lib.operation import Sum, Dot
def train_sample(self, epochs=200):
    for epoch in range(epochs):
        with Graph() as g:
            x = Tensor(self.X, name='X')
            y = Tensor(self.y, name='y')
            weights = Tensor(self.W, name='W')
    
            z = Sum((Dot(x, weights) + (-y)) ** Constant(2))
            path, vis = g.compute_path(z.obj_id)
            executor = Execution(path)
            executor.forward()
            self.loss = z()
    
            print(f"Loss: {self.loss}")
            executor.backward_ad()
            self.W = self.W - 0.001*weights.grad
````
