import torch
import gpytorch
from gpytorch.kernels import RBFKernel, CosineKernel


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel,likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Create a toy dataset
train_x = torch.linspace(0, 1, 10)
train_y = torch.sin(train_x * (2 * torch.Tensor([3.14])))

# initialize likelihood and model

# Define two kernels
rbf_kernel = RBFKernel()
rbf_kernel.lengthscale = 0.1
cosine_kernel = CosineKernel()
cosine_kernel.period_length = 0.2
# Define a kernel that is a combination of the two kernels
combined_kernel = rbf_kernel + cosine_kernel
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, combined_kernel,likelihood)

# Train the model
# model.train()
# Access the lengthscale and variance parameters of the combined kernel
print(list(rbf_kernel.parameters()))
# for name, param in rbf_kernel.named_parameters():
    # print(name, param)
print("#############################")
print(list(cosine_kernel.parameters()))
# for name, param in cosine_kernel.named_parameters():
    # print(name, param)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# for name, param in combined_kernel.named_parameters():
    # print(name, param)
print(list(combined_kernel.parameters()))
print("ALPHAA")
cosine_kernel.period_length = 0.4
print(list(cosine_kernel.parameters()))
print(list(combined_kernel.parameters()))
print("BETAA")
rbf_kernel.lengthscale = 0.9
print(list(rbf_kernel.parameters()))
print(list(combined_kernel.parameters()))
# Output:
# [Parameter containing:
# tensor([0.6931], requires_grad=True),
# Parameter containing:
# tensor([-0.6931], requires_grad=True)]