

# Simple Artificial Neural Network in PyTorch

This repo contains five implementations of a neural network going from low to high levels of abstraction.

- **L1Tensor.py** only uses PyTorch tensor parameter, computing the loss, gradient descent and model explicitly.

- **L2Loss.py** is indentical to L1Tensor.py but with pre-defined loss from torch.functional.

- **L3LossOptim.py** is idential to L2Loss.py but with pre-defined stochastic gradient descent optimisation from torch.optim.

- **L4LossOptimLinear.py** is idential to L3LossOptim.py but with pre-defined Linear layer from torch.nn.

- **L5Sequential.py** is idential to L4LossOptimLinear.py but with pre-defined sequential neural network from torch.nn.


# Parameter Partitioner Program

I wrote this program to provide helper functions to partition the parameters of large models. This is indended to be used when different parameter groups need different initialisation or optimisation.