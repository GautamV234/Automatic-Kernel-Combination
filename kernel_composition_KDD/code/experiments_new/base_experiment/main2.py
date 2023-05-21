##############################
# Importing libraries
##############################
from kernel_composition_KDD.code.model_new.model_gpy import Best_model_params, Inference, KernelCombinations
import numpy as np
import matplotlib.pyplot as plt
import psutil
import yaml
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
import time
import GPy
import typing
import gpytorch
import torch
from jax.config import config
import jax.numpy as jnp
import optax as ox
import jax.random as jr
import matplotlib.pyplot as plt
import gpytorch.kernels as gtk
import GPy.kern as gpk
from tabulate import tabulate

# from jaxutils import Dataset
# import jaxkern as jk

# OBJECTIVE - TRYING TO GET THE SAME VALUES OF 
# HYPERPARAMETERS IF WE GIVE THE EXACT SAME STRUCTURE KERNEL (COMPLEX) 
# ALREADY TO THE MODELS IN BOTH GPY AND GPYTORCH 

class covar_kernels():
    def __init__(self, kernel_list:typing.List[str],num_features=1,device='cpu'):
        self.kernel_list = kernel_list
        self.num_features = num_features
        self.device = device

    def get_base_kernels(self):
        all_kernels= {'RBF':GPy.kern.RBF(input_dim=self.num_features),
                    'Matern32':GPy.kern.Matern32(input_dim=self.num_features),
                    'Matern52':GPy.kern.Matern52(input_dim=self.num_features),
                    'RQ':GPy.kern.RatQuad(input_dim=self.num_features),
                    'Linear':GPy.kern.Linear(input_dim=self.num_features),
                    'Periodic':GPy.kern.PeriodicExponential(input_dim=self.num_features),
                    'Cosine':GPy.kern.Cosine(input_dim=self.num_features)}
        base_kernels = {}
        for kernel_name in self.kernel_list:
            assert kernel_name in all_kernels.keys(), f"{kernel_name} is not a valid kernel name"
            for i in range(self.num_features):
                if (kernel_name =='RBF'):
                    kernel = GPy.kern.RBF(input_dim=1,active_dims=[i])
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Matern32'):
                    kernel = GPy.kern.Matern32(input_dim=1,active_dims=[i])
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Matern52'):
                    kernel = GPy.kern.Matern52(input_dim=1,active_dims=[i])
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='RQ'):
                    kernel = GPy.kern.RatQuad(input_dim=1,active_dims=[i])
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Linear'):
                    kernel = GPy.kern.Linear(input_dim=1,active_dims=[i])
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Periodic'):
                    kernel = GPy.kern.PeriodicExponential(input_dim=1,active_dims=[i])
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Cosine'):
                    kernel = GPy.kern.Cosine(input_dim=1,active_dims=[i])
                    base_kernels[f"{kernel_name}_{i}"] = kernel
        return base_kernels


#########################
# Data Loading
#########################
def data_loader(data_path):
    data = sio.loadmat(data_path)
    X = data['X']
    y = data['y']
    X = np.array(X).reshape(-1)
    y = np.array(y).reshape(-1)
    return X, y





if __name__ == '__main__':
    process = psutil.Process()
    pid = process.pid
    print(f"Process ID : {pid}")
    # print device name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    ######################################
    # Defining paths
    #####################################
    print(os.getcwd())
    root_path = os.path.join(os.getcwd(), "kernel_composition_KDD","code","experiments_new", "base_experiment")
    # root_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments/mauna_loa_extrapolate_lbfgs'
    data_path = os.path.join(root_path, "kernel_composition_KDD","data", "mauna2011.mat")
    # data_path = '/home/gautam.pv/nlim/kernel_composition_KDD/data/mauna2011.mat'
    yaml_file = os.path.join(os.getcwd(), "kernel_composition_KDD","yaml", "new_synthetic.yaml")
    ######################################
    # Data Loading
    ######################################
    # X, y = data_loader(data_path)
    # get the train and test data by keeping the last 20% of the data as test data
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # create an extrapolation dataset for sine function by adding noise
    # Create synthetic training data from sampling from a fixed kernel combination
    fixed_kernel_combination = (gpk.RBF(1,0.7,3)*gpk.Matern32(1,0.5,2))+gpk.Linear(1,0.3)
    _ = np.random.uniform(-1,1,(1,1))
    fixed_model = GPy.models.GPRegression(_, _, fixed_kernel_combination)
    # make a GPy model using this kernel
    N = 1000
    train_n = int(0.8*N)
    X = np.linspace(-10, 13, N).reshape(-1, 1)
    Y = fixed_model.posterior_samples_f(X, size=1).reshape(-1)
    # break into X_train and X_test
    X_train = X[:int(0.8*N)]
    X_test = X[int(0.8*N):]
    Y_train = Y[:int(0.8*N)]
    Y_test = Y[int(0.8*N):]
    ##############
    # GPy Model
    ##############
    gpy_kernel = (gpk.RBF(input_dim=1, lengthscale=1)*gpk.Matern32(input_dim=1, lengthscale=1))+ gpk.Linear(input_dim=1)
    gpy_model = GPy.models.GPRegression(X_train, Y_train.reshape(-1,1), gpy_kernel)
    gpy_loss = - gpy_model.log_likelihood()
    ##############
    # GPyTorch Model
    ##############
    class GPRegression_model(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegression_model, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gtk.ScaleKernel(gtk.RBFKernel()*gtk.MaternKernel(nu=1.5) + gtk.LinearKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    gpytorch_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gpytorch_likelihood.noise = gpy_model.likelihood.variance[0]
    gpytorch_model = GPRegression_model(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(), gpytorch_likelihood)
    output = gpytorch_model(torch.from_numpy(X_train).float())
    # MLL
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpytorch_likelihood, gpytorch_model)
    gpytorch_loss = - mll(output, torch.from_numpy(Y_train).float())
    training_iter = 1000
    print(f"Number of data points : {N}")
    print(f"Number of training iterations : {training_iter}")
    print(f"Initial GPy Loss : {gpy_loss}")
    print(f"Initial GPyTorch Loss: {gpytorch_loss.item()*train_n}")
    ##################
    # OPTIMIZATION 
    ##################
    # GPY MODEL
    # gpy_model.optimize(max_iters=training_iter)
    gpy_model.optimize_restarts(num_restarts=10, max_iters=training_iter)
    # Gpytorch model
    gpytorch_model.train()
    gpytorch_likelihood.train()
    # Train using Adam
    optimizer = torch.optim.Adam(gpytorch_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpytorch_likelihood, gpytorch_model)
    # Training loop
    for i in range(training_iter):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = gpytorch_model(torch.from_numpy(X_train).float())
        # Calc loss and backprop gradients
        loss = -mll(output, torch.from_numpy(Y_train).float())
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
    
    print('Final GPy loss: {}'.format(-gpy_model.log_likelihood()))
    print('Final gpytorch loss: {}'.format(train_n* -mll(output, torch.from_numpy(Y_train).float()).item()))
    print("#############################################")
    print("TRUE PARAMS")
    print(fixed_model)
    print("GPY PARAMS")
    print(gpy_model)        
    # # Print the learnt GPy parameters
    # print('GPy lengthscale: {}'.format(gpy_model.kern.lengthscale[0]))
    # print('GPy variance: {}'.format(gpy_model.kern.variance[0]))
    # print('GPy noise: {}'.format(gpy_model.likelihood.variance[0]))
    print("#############################################")
    # print("GPYTORCH PARAMS")
    # Get the parameters of the gpytorch model for each component kernel in the additive kernel

    # # Iterate over the named parameters and extract the kernel name
    # for name, param in gpytorch_model.named_parameters():
    #     if param.requires_grad:
    #         # Split the parameter name into its components
    #         print(f"name : {name}")
    #         components = name.split(".")
    #         # Check if this is a kernel parameter
    #         if components[0] == "covar_module" and components[-1] == "raw_lengthscale":
    #             # Extract the kernel name from the parameter name
    #             kernel_idx = [int(c) for c in components[3:-1] if c.isdigit()]
    #             print(f"kernel_idx:{kernel_idx}")
    #             x = gpytorch_model.covar_module.base_kernel.sub_kernels()
    #             for i in x:
    #                 print(type(i).__name__)

    #             elem = None
    #             for idx in kernel_idx:
    #                 print(f"idx is {idx}")
    #                 print(gpytorch_model.covar_module.base_kernel['AdditiveKernel'])
    #                 if not elem:
    #                     elem = gpytorch_model.covar_module.base_kernel[idx]
    #                     print(f"elem becomes {type(elem).__name__}")
    #                 else:
    #                     elem = elem[idx]
    #                     print(f"elem becomes {type(elem).__name__}")
    #             print(f"elem is {type(elem).__name__}")
    #             print(f"OG elem is {type(gpytorch_model.covar_module.base_kernel[kernel_idx]).__name__}")
    #             kernel_name = kernel_names[type(gpytorch_model.covar_module.base_kernel[kernel_idx]).__name__]
    #             # Print the kernel name and the parameter value
    #             actual_lengthscale = torch.exp(param).item()
    #             print(f"{kernel_name} lengthscale: {actual_lengthscale}")
    #         elif components[0]=="covar_module" and components[-1]=="raw_variance":
    #             # Extract the kernel name from the parameter name
    #             kernel_idx = [int(c) for c in components[3:-1] if c.isdigit()]
    #             print(f"kernel_idx: {kernel_idx}")
    #             print(gpytorch_model.covar_module.base_kernel)
    #             elem = None
    #             for idx in kernel_idx:
    #                 if not elem:
    #                     elem = gpytorch_model.covar_module.base_kernel[idx]
    #                 else:
    #                     elem = elem[idx]
    #             print(f"elem is {type(elem).__name__}")
    #             print(f"OG elem is {type(gpytorch_model.covar_module.base_kernel[kernel_idx]).__name__}")
    #             kernel_name = kernel_names[type(gpytorch_model.covar_module.base_kernel[kernel_idx]).__name__]
    #             actual_variance = torch.exp(param).item()
    #             # Print the kernel name and the parameter value
    #             print(f"{kernel_name} variance: {actual_variance}")
    #         elif components[0]=="covar_module" and components[-1]=="raw_outputscale":
    #             print(f"covar_module Outputscale: {torch.exp(param).item()}")
    #         elif components[0]=="likelihood" and components[-1]=="raw_noise":
    #             print(f"Noise: {torch.exp(param).item()}")
        # print(gpytorch_model)
        # for name, param in gpytorch_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.data}")
    # print("#############################################")
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
# Get the GPy predictions
gpy_mean, gpy_var = gpy_model.predict(X_test)

# Get the gpytorch predictions
gpytorch_model.eval()
gpytorch_likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = gpytorch_likelihood(gpytorch_model(torch.from_numpy(X_test).float()))

axs[0].plot(X_train, Y_train, 'kx', mew=2)
axs[0].plot(X_test, gpy_mean, 'b', lw=2, label='GPy mean')
axs[0].plot(X_test,Y_test, 'g', lw=2, label='True mean')
plt.suptitle(f'GPy vs gpytorch on Synthetic Data \n (gpk.RBF(1,0.7,3)*gpk.Matern32(1,0.5,2))+gpk.Linear(1,0.3)')
axs[0].set_title(f'GPy (Optimized Restarts) (Num iteration: {training_iter})')
axs[1].set_title(f'Gpytorch (Num iteration: {training_iter})')
axs[1].plot(X_train, Y_train, 'kx', mew=2)
axs[1].plot(X_test,Y_test, 'g', lw=2, label='True mean')
axs[1].plot(X_test, observed_pred.mean.numpy(), 'r', lw=2, label='gpytorch mean')
axs[0].fill_between(X_test.flatten(), gpy_mean.flatten() - 2 * np.sqrt(gpy_var.flatten()), gpy_mean.flatten() + 2 * np.sqrt(gpy_var.flatten()), alpha=0.5, color='blue', label='GPy uncertainty')
# Get the lower and upper confidence bounds for the gpytorch model
lower, upper = observed_pred.confidence_region()
axs[1].fill_between(X_test.flatten(), lower.detach().numpy().flatten(), upper.detach().numpy().flatten(), alpha=0.5, color='red', label='gpytorch uncertainty')
fig.legend(loc='upper left')
# save the figure
base_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments_new/base_experiment'
path = os.path.join(base_path, f'gpy_gpytorch_test_{training_iter}_{N}_DP_OPT.png')
plt.savefig(path)
