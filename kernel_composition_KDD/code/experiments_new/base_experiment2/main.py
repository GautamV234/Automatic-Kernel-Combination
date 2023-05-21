##############################
# Importing libraries
##############################
from kernel_composition_KDD.code.model_new.model_gpy import Best_model_params, Inference, KernelCombinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import psutil
import yaml
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
import time
import GPy
import shutil
import gpytorch
import torch
from jax.config import config
import jax.numpy as jnp
import optax as ox
from tqdm import tqdm
import jax.random as jr
import matplotlib.pyplot as plt
import gpytorch.kernels as gtk
import GPy.kern as gpk
# from tabulate import tabulate
import imageio

# from jaxutils import Dataset
# import jaxkern as jk

# OBJECTIVE - TRYING TO GET THE SAME VALUES OF 
# HYPERPARAMETERS IF WE GIVE THE EXACT SAME STRUCTURE KERNEL (COMPLEX) 
# ALREADY TO THE MODELS IN BOTH GPY AND GPYTORCH *(NOW USING GPU)*

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
    root_path = os.path.join(os.getcwd(), "kernel_composition_KDD","code","experiments_new", "base_experiment2")
    ######################################
    # Data Loading
    ######################################
    fixed_kernel_combination = (gpk.RBF(1,0.7,3)*gpk.Matern32(1,0.5,2))+gpk.Linear(1,0.3)
    _ = np.random.uniform(-1,1,(1,1))
    fixed_model = GPy.models.GPRegression(_, _, fixed_kernel_combination)
    # make a GPy model using this kernel
    ##############################
    # HYPERPARAMETERS
    N = 20
    training_iter = 100
    ##############################
    train_n = int(0.8*N)
    X = np.linspace(-10, 13, N).reshape(-1, 1)
    Y = fixed_model.posterior_samples_f(X, size=1).reshape(-1)
    # break into X_train and X_test
    x_train_csv,y_train_csv = pd.read_csv(os.path.join(root_path, f"train_data_{N}.csv"), usecols=[1,2], header=0).values.T
    x_test_csv,y_test_csv = pd.read_csv(os.path.join(root_path, f"test_data_{N}.csv"), usecols=[1,2], header=0).values.T
    X_train = x_train_csv.reshape(-1,1)
    X_test = x_test_csv.reshape(-1,1)
    Y_train = y_train_csv.reshape(-1)
    Y_test = y_test_csv.reshape(-1)
    ############
    # GPy Model
    ############
    # gpy_kernel = (gpk.RBF(input_dim=1, lengthscale=1)*gpk.Matern32(input_dim=1, lengthscale=1))+ gpk.Linear(input_dim=1)
    gpy_kernel = (gpk.RBF(1,0.7,3)*gpk.Matern32(1,0.5,2))+gpk.Linear(1,0.3)
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
    # gpytorch_likelihood.noise = gpy_model.likelihood.variance[0]
    gpytorch_model = GPRegression_model(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(), gpytorch_likelihood)
    gpytorch_model = gpytorch_model.cuda()
    gpytorch_likelihood = gpytorch_likelihood.cuda()
    X_train_cuda = torch.from_numpy(X_train).float().cuda()
    Y_train_cuda = torch.from_numpy(Y_train).float().cuda()
    X_test_cuda = torch.from_numpy(X_test).float().cuda()
    Y_test_cuda = torch.from_numpy(Y_test).float().cuda()
    output = gpytorch_model(X_train_cuda)
    # MLL
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpytorch_likelihood, gpytorch_model)
    gpytorch_loss = - mll(output, Y_train_cuda)
    print(f"Number of data points : {N}")
    print(f"Number of training iterations : {training_iter}")
    print(f"Initial GPy Loss : {gpy_loss}")
    print(f"Initial GPyTorch Loss: {gpytorch_loss.item()*train_n}")
    ##################
    # OPTIMIZATION 
    ##################
    ############################## 
    # GPY MODEL
    ##############################
    print(f"Optimizing GPy model...")
    gpy_model.optimize(max_iters=training_iter)
    print(f"Optimizing GPyTorch model...")
    # gpy_model.optimize_restarts(num_restarts=10, max_iters=training_iter)
    ##############################
    # Gpytorch model
    ##############################
    gpytorch_model.train()
    gpytorch_likelihood.train()
    # Train using Adam
    optimizer = torch.optim.Adam(gpytorch_model.parameters(), lr=0.1)  # Includes Gaussian Likelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpytorch_likelihood, gpytorch_model)
    # Training loop
    
    dir_path = os.path.join(root_path,"plots")
    if os.path.exists(os.path.join(root_path,"plots")):
            shutil.rmtree(dir_path)
            # os.rmdir(os.path.join(root_path,"plots"))
    os.mkdir(os.path.join(root_path,"plots"))
    for i in tqdm(range(training_iter)):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = gpytorch_model(X_train_cuda)
        # Calc loss and backprop gradients
        loss = -mll(output, Y_train_cuda)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
        gpytorch_model.eval()
        gpytorch_likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            fig_v, ax_v = plt.subplots()
            observed_pred = gpytorch_likelihood(gpytorch_model(X_train_cuda))
            ax_v.set_title(f'Iteration {i+1} GPyTorch Loss: {loss.item()*train_n:.3f} \n Data points = {N} ')
            ax_v.plot(X_train, Y_train, color = 'b',label = 'True mean (TRAIN)')
            ax_v.plot(X_test,Y_test, color = 'g', label='True mean (TEST)')
            ax_v.plot(X_train, observed_pred.mean.cpu().numpy().reshape(-1,1), color = 'r', lw=2, label='gpytorch mean (TRAIN)')
            # plot the testing data
            # ax_v.plot(X_test, Y_test, 'kx', mew=2,color = 'g', label='True mean (TEST)')
            test_prediction = gpytorch_likelihood(gpytorch_model(X_test_cuda))
            ax_v.plot(X_test, test_prediction.mean.cpu().numpy().reshape(-1,1), color = 'y', lw=2, label='gpytorch mean (TEST)')
            lower_test, upper_test = test_prediction.confidence_region()
            lower_train, upper_train = observed_pred.confidence_region()
            ax_v.fill_between(X_train.flatten(), lower_train.detach().cpu().numpy().flatten(), upper_train.detach().cpu().numpy().flatten(), alpha=0.5, color='blue', label='model uncertainty (TRAIN)')
            ax_v.fill_between(X_test.flatten(), lower_test.detach().cpu().numpy().flatten(), upper_test.detach().cpu().numpy().flatten(), alpha=0.5, color='orange', label='model  uncertainty (TEST)')
            save_path = os.path.join(root_path, 'plots', f'plots_{i+1}.png')
            # show labels on top left
            ax_v.legend(loc='upper right')
            fig_v.savefig(save_path)
        gpytorch_model.train()
        gpytorch_likelihood.train()    
    image_folder = os.path.join(root_path, 'plots')
    video_name = f'Predictions_{N}_{training_iter}.mp4'
    video_name = os.path.join(root_path, 'videos', video_name)
    image_files = [f'{image_folder}/plots_{i}.png' for i in range(1, training_iter)]
    with imageio.get_writer(video_name, mode='I',fps=10) as writer:
        for  image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)
    print(f"Video stored in {video_name}")
    print("#############################################")
    print('Final GPy loss: {}'.format(-gpy_model.log_likelihood()))
    print('Final gpytorch loss: {}'.format(train_n* -mll(output, Y_train_cuda).item()))
    print("#############################################")
    print("TRUE PARAMS")
    print(fixed_model)
    print("GPY PARAMS")
    print(gpy_model)        
    print("#############################################")
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    # Get the GPy predictions
    gpy_mean, gpy_var = gpy_model.predict(X_test)

    # Get the gpytorch predictions
    gpytorch_model.eval()
    gpytorch_likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = gpytorch_likelihood(gpytorch_model(X_test_cuda))

    plt.suptitle(f'GPy vs gpytorch({device}) on Synthetic Data \n (gpk.RBF(1,0.7,3)*gpk.Matern32(1,0.5,2))+gpk.Linear(1,0.3)')
    # GPy Plot
    axs[0].plot(X_train, Y_train, 'kx', mew=2)
    axs[0].plot(X_test, gpy_mean, 'b', lw=2, label='GPy mean')
    axs[0].plot(X_test,Y_test, 'g', lw=2, label='True mean')
    axs[0].fill_between(X_test.flatten(), gpy_mean.flatten() - 2 * np.sqrt(gpy_var.flatten()), gpy_mean.flatten() + 2 * np.sqrt(gpy_var.flatten()), alpha=0.5, color='blue', label='GPy uncertainty')
    axs[0].set_title(f'GPy (Num iteration: {training_iter}) (DP : {N})')
    # gpytorch plot
    axs[1].set_title(f'Gpytorch (Num iteration: {training_iter}) (DP : {N})')
    axs[1].plot(X_train, Y_train, 'kx', mew=2)
    axs[1].plot(X_test,Y_test, 'g', lw=2, label='True mean')
    axs[1].plot(X_test, observed_pred.mean.cpu().numpy(), 'r', lw=2, label='gpytorch mean')
    lower, upper = observed_pred.confidence_region()
    axs[1].fill_between(X_test.flatten(), lower.detach().cpu().numpy().flatten(), upper.detach().cpu().numpy().flatten(), alpha=0.5, color='red', label='gpytorch uncertainty')
    fig.legend(loc='upper left')
    # save the figure
    base_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments_new/base_experiment2'
    path = os.path.join(base_path, f'gpy_gpytorch_GPU_test_{training_iter}_{N}_DP.png')
    plt.savefig(path)
    print(f"Figure saved in {path}")