import torch
import gpytorch
import numpy as np
import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt
import gpytorch
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler

#################
# TESTING IF L-BFGS WORKS
##################
# Define the GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # inner_kernel = (gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.RQKernel())*gpytorch.kernels.PeriodicKernel()
        # self.covar_module = gpytorch.kernels.ScaleKernel(inner_kernel)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def data_loader(data_path):
    data = sio.loadmat(data_path)
    X = data['X']
    y = data['y']
    X = np.array(X).reshape(-1)
    y = np.array(y).reshape(-1)
    return X, y

root_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments/mauna_loa_extrapolate_lbfgs'
data_path = '/home/gautam.pv/nlim/kernel_composition_KDD/data/mauna2011.mat'
    
X, y = data_loader(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move data and model to the GPU
X_train = torch.from_numpy(X_train).float().squeeze().to(device)
Y_train = torch.from_numpy(y_train).float().squeeze().to(device)
X_test = torch.from_numpy(X_test).float().squeeze().to(device)
Y_test = torch.from_numpy(y_test).float().squeeze().to(device)
train_x = X_train.to(device)
train_y = Y_train.to(device)
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = ExactGPModel(train_x, train_y, likelihood).to(device)

# Train the model using the L-BFGS optimizer
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
training_iterations = 100
def closure():
    optimizer.zero_grad()
    output = model(train_x)
    print(f"output: {output}")
    loss = -output.log_prob(train_y).sum()
    print(f"Loss: {loss.item()}")
    loss.backward()
    return loss
for i in range(training_iterations):
    optimizer.step(closure)
