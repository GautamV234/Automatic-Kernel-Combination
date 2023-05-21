##############################
# Importing libraries
##############################
from kernel_composition_KDD.code.model_new.model_gpy import Best_model_params, Inference, KernelCombinations
import numpy as np
import pandas as pd
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
    np.random.seed(123)
    fixed_kernel_combination = (gpk.RBF(1,0.7,3)*gpk.Matern32(1,0.5,2))+gpk.Linear(1,0.3)
    _ = np.random.uniform(-1,1,(1,1))
    fixed_model = GPy.models.GPRegression(_, _, fixed_kernel_combination)
    # make a GPy model using this kernel
    N = 20
    train_n = int(0.8*N)
    X = np.linspace(-10, 13, N).reshape(-1, 1)
    Y = fixed_model.posterior_samples_f(X,full_cov=True, size=1).reshape(-1)
    # break into X_train and X_test
    X_train = X[:int(0.8*N)]
    X_test = X[int(0.8*N):]
    Y_train = Y[:int(0.8*N)]
    Y_test = Y[int(0.8*N):]
    train_data = {'X_train': X_train.reshape(-1), 'Y_train': Y_train.reshape(-1)}
    test_data = {'X_test': X_test.reshape(-1), 'Y_test': Y_test.reshape(-1)}
    # save the data
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    train_df.to_csv(os.path.join(root_path, f"train_data_{N}.csv"))
    test_df.to_csv(os.path.join(root_path, f"test_data_{N}.csv"))
    # fetch the data from the csv file
    x_train_csv,y_train_csv = pd.read_csv(os.path.join(root_path, f"train_data_{N}.csv"), usecols=[1,2], header=0).values.T
    x_test_csv,y_test_csv = pd.read_csv(os.path.join(root_path, f"test_data_{N}.csv"), usecols=[1,2], header=0).values.T
    plt.plot(x_train_csv, y_train_csv, 'b.')
    plt.plot(x_test_csv, y_test_csv, 'r.')
    # put points after every 2 points on X axis
    plt.xticks(np.arange(-10, 13, 2))
    # add title and axis names
    plt.title(f'Synthetic Data for N = {N}')
    plt.xlabel('X')
    plt.ylabel('(gpk.RBF(1,0.7,3)*gpk.Matern32(1,0.5,2))+gpk.Linear(1,0.3)')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(root_path,f"synthetic_data_{N}.png"))
    ######################################