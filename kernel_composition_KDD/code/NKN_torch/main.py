import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
import torch.nn as nn

prim_kernels = [RBFKernel(), RBFKernel(), RBFKernel(),RBFKernel()]
scaling_factor = 10
class HiddenLayer(nn.Module):
    def __init__(self,hiddens=6):
        super(HiddenLayer, self).__init__()
        self.log_fc1 = nn.Parameter(torch.Tensor(len(prim_kernels), hiddens))
        self.reset_parameters()

    def forward(self, input):
        return nn.functional.linear(input, self.log_fc1.exp())
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_fc1)

if __name__ == '__main__':
    train_x = torch.rand(10, 1)
    train_y = torch.sin(train_x)
    layer1 = HiddenLayer(hiddens=6)
    layer2 = HiddenLayer(hiddens=1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()