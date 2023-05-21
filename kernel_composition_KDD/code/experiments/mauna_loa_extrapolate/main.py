##############################
# Importing libraries
##############################
import typing
from kernel_composition_KDD.code.model.model_3 import Best_model_params, Inference, KernelCombinations
import numpy as np
import torch
import matplotlib.pyplot as plt
import gpytorch
import psutil
import yaml
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
import time
import torch

class covar_kernels():
    def __init__(self, kernel_list:typing.List,num_features=1,device='cpu'):
        self.kernel_list = kernel_list
        self.num_features = num_features
        self.device = device

    def get_base_kernels(self):
        all_kernels = {'RBF':gpytorch.kernels.RBFKernel().to(self.device),
        'Matern12': gpytorch.kernels.MaternKernel(nu=0.5).to(self.device),
        'Matern32': gpytorch.kernels.MaternKernel(nu=1.5).to(self.device),
        'Matern52': gpytorch.kernels.MaternKernel(nu=2.5).to(self.device),
        'RQ': gpytorch.kernels.RQKernel().to(self.device),
        'Linear': gpytorch.kernels.LinearKernel().to(self.device),
        'Periodic': gpytorch.kernels.PeriodicKernel().to(self.device),
        'Cosine': gpytorch.kernels.CosineKernel().to(self.device)}
        base_kernels = {}
        for kernel_name in self.kernel_list:
            assert kernel_name in all_kernels.keys(), f"{kernel_name} is not a valid kernel name"
            for i in range(self.num_features):
                if (kernel_name =='RBF'):
                    kernel = gpytorch.kernels.RBFKernel(active_dims=[i]).to(self.device)
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Matern12'):
                    kernel = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=[i]).to(self.device)
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Matern32'):
                    kernel = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=[i]).to(self.device)
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Matern52'):
                    kernel = gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[i]).to(self.device)
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='RQ'):
                    kernel = gpytorch.kernels.RQKernel(active_dims=[i]).to(self.device)
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Linear'):
                    kernel = gpytorch.kernels.LinearKernel(active_dims=[i]).to(self.device)
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Periodic'):
                    kernel = gpytorch.kernels.PeriodicKernel(active_dims=[i]).to(self.device)
                    base_kernels[f"{kernel_name}_{i}"] = kernel
                elif (kernel_name =='Cosine'):
                    kernel = gpytorch.kernels.CosineKernel(active_dims=[i]).to(self.device)
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

#########################
# KL Divergence
#########################
def compute_kl_divergence(p, q):
    """Computes the KL divergence between two distributions."""
    kl_divergence = gpytorch.distributions.kl_divergence(p, q)
    

def plot_scores(store_path,scores_dict):
    """Plots the different scores for the model built sequentially."""
    print(scores_dict)
    # plot the scores
    mean_sqared_errors = []
    negative_log_predictive_density = []
    mean_standardized_log_loss = []
    quantile_coverage_error = []
    root_mean_squared_error = []
    model_names = []
    for model_name, scores in scores_dict.items():
        mean_sqared_errors.append(scores["Mean Squared Error"])
        negative_log_predictive_density.append(scores["Negative Log Predictive Density"])
        mean_standardized_log_loss.append(scores["Mean Standardized Log Loss"])
        quantile_coverage_error.append(scores["Quantile Coverage Error"])
        root_mean_squared_error.append(scores["Root Mean Squared Error"])
        model_names.append(model_name)
    # plot all the scores as independent plots
    fig, axs = plt.subplots(5, 1, figsize=(20, 20))
    axs[0].plot(model_names, mean_sqared_errors)
    axs[0].set_title("Mean Squared Error")
    # annotate the points in the plot with the model names
    axs[0].set_xticks(model_names)
    axs[1].plot(model_names, negative_log_predictive_density)
    axs[1].set_title("Negative Log Predictive Density")
    axs[1].set_xticks(model_names)
    axs[2].plot(model_names, mean_standardized_log_loss)
    axs[2].set_title("Mean Standardized Log Loss")
    axs[2].set_xticks(model_names)
    axs[3].plot(model_names, quantile_coverage_error)
    axs[3].set_title("Quantile Coverage Error")
    axs[3].set_xticks(model_names)
    axs[4].plot(model_names, root_mean_squared_error)
    axs[4].set_title("Root Mean Squared Error")
    axs[4].set_xticks(model_names)
    # beautify the x-labels
    plt.setp(axs, xticks=model_names, xticklabels=model_names)
    # plt.setp(axs, yticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # plt.setp(axs, yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.setp(axs, xlabel="Model Names")
    plt.setp(axs, ylabel="Scores")
    plt.tight_layout()
    # beautify the plots
    # plt.subplots_adjust(top=0.92)
    plt.suptitle('Scores for different models', fontsize=14, fontweight='bold')        
    plt.savefig(os.path.join(store_path, "scores.png"))




if __name__ == '__main__':
    process = psutil.Process()
    pid = process.pid
    print(f"Process ID : {pid}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    ######################################
    # Defining paths
    #####################################
    root_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments/mauna_loa_extrapolate'
    data_path = '/home/gautam.pv/nlim/kernel_composition_KDD/data/mauna2011.mat'
    yaml_file = '/home/gautam.pv/nlim/kernel_composition_KDD/yaml/mauna_loa_extrapolate.yaml'
    ######################################
    # Data Loading
    ######################################
    X, y = data_loader(data_path)
    # get the train and test data by keeping the last 20% of the data as test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    ##########################
    # Convert to torch tensors
    ##########################
    X_train = torch.from_numpy(X_train).float().squeeze().to(device)
    Y_train = torch.from_numpy(y_train).float().squeeze().to(device)
    X_test = torch.from_numpy(X_test).float().squeeze().to(device)
    Y_test = torch.from_numpy(y_test).float().squeeze().to(device)
    ##########################
    # Kernel Combination Initialization
    ##########################
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)
        for id, specs  in yaml_data.items():
            start = time.time()
            print(f"Experiment ID : {id}")
            exp_path = os.path.join(root_path, f"exp_{id}")
            if not os.path.exists(exp_path):
                os.mkdir(exp_path)
            heuristic = specs['heuristic']
            if heuristic:
                ckpt = not specs['ckpt']
                ckpt_path = specs['ckpt_path']
                kernels_yaml = specs['kernels']
                combination_list = specs['combination_list']
                degree = sum(combination_list)
                combination_lists = [combination_list]
                degrees_ = [degree]
                num_epochs = specs['num_epochs']
                lr = specs['lr']
                stopping = specs['stopping']
                log_file_name = os.path.join(exp_path, f"heuristic_log.txt")
                metrics_log_file_name = os.path.join(exp_path, f"heuristic_metrics.txt")
                for degree_val, combination_list in zip(degrees_, combination_lists):
                    base_kernels_map = covar_kernels(kernel_list = kernels_yaml,device=device).get_base_kernels()
                    print(f"Required Degree : {degree_val}")
                    print(f"Combination Used : {combination_list}")
                    kc1 = KernelCombinations(degree_val, base_kernels_map, X_train, Y_train, X_test, Y_test,
                                            ckpt_path=ckpt_path, device=device, reinitialize=ckpt, num_initializations=1,num_epochs=num_epochs, lr=lr,stopping=stopping)
                    log_file = open(log_file_name, "w")
                    best_model, progressive_models = kc1.main(
                        heuristic=True, combination_list=combination_list, log_file=log_file)
                    log_file.close()
                    # Inference
                    metrics_log_file = open(metrics_log_file_name,"w")
                    scores_visualizer = {}
                    for progressive_model_params in progressive_models:
                        progressive_model_params: Best_model_params
                        inference = Inference(X_train, Y_train, X_test, Y_test,
                                            progressive_model_params, degree=degree_val, device=device)
                        print(
                            f"Performing Inference for {degree_val}: {progressive_model_params.combination_name}")
                        inference.plot_fit_gpytorch(figure_save_path=exp_path)
                        inference.compute_test_predictions()
                        scores_dict = inference.tabulate_scores(data="test",metrics_log_file=metrics_log_file)
                        scores_visualizer[progressive_model_params.combination_name] = scores_dict
                    # Plotting Scores
                    plot_scores(exp_path,scores_visualizer)
                    end = time.time()
                    print(f"Time taken for experiment {id} : {end-start}",file=metrics_log_file)
                    metrics_log_file.close()
                            