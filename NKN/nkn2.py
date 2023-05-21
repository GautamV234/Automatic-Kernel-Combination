import math
import torch
import gpytorch

# Defining the neural network
class KernelNetwork(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(KernelNetwork, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        matern32_kernel  = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        lin_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        self.rbf_kernel = rbf_kernel        
        self.matern32_kernel = matern32_kernel
        self.lin_kernel = lin_kernel
        n = 3
        # self.hidden_values = torch.nn.Parameter(torch.rand(n) * (n-1) + 1)
        # self.hidden_values = [torch.nn.Parameter(torch.rand(1) * (n-1) + 1, requires_grad=True) for i in range(n)]
        self.hidden_values = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(1) * (n-1) + 1, requires_grad=True) for i in range(n)])
        # self.hidden_values = torch.nn.Parameter(torch.rand(n, 1) * (n-1) + 1, requires_grad=True)
        # self.hidden_values_tensor = torch.stack(self.hidden_values)
        

        self.mean_module = gpytorch.means.ConstantMean()
        
    def forward(self, x):
        rbf_out = self.rbf_kernel(x).evaluate()
        matern32_out = self.matern32_kernel(x).evaluate()
        lin_out = self.lin_kernel(x).evaluate()
        n = 3
        tensors = [rbf_out, matern32_out, lin_out]
        covar_x= sum([torch.round(self.hidden_values[i]) *  tensors[i] for i in range(n)]) 
        # covar_x = sum([torch.round(self.hidden_values[i][0]) * tensors[i] for i in range(n)])
        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(model, train_x, train_y, optimizer, likelihood):
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)    
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    for i in range(len(model.hidden_values)):
        print(f"Gradient for hidden scalar {i}: {model.hidden_values[i].grad}")
    optimizer.step()
    return loss.item()

########
# Training data
########
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi))
train_y += 0.1 * torch.randn_like(train_y)

# Initialize the model and likelihood
model = KernelNetwork(train_x, train_y)
likelihood = model.likelihood

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
num_epochs = 100
for i in range(num_epochs):
    loss = train(model, train_x, train_y, optimizer, likelihood)
    print(f"Epoch {i+1}/{num_epochs}: Loss = {loss:.3f}")
    # Print the final values of the hidden scalars
    print("Final hidden scalars:", model.hidden_values)

# Print the final values of the hidden scalars
print("Final hidden scalars:", model.hidden_values.data)
