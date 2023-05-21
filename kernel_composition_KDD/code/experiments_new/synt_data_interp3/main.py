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
                    kernel = GPy.kern.StdPeriodic(input_dim=1,active_dims=[i])
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


def plot_scores(store_path,scores_dict):
    """Plots the different scores for the model built sequentially."""
    print(scores_dict)
    negative_log_predictive_density = []
    root_mean_squared_error = []
    model_names = []
    for model_name, scores in scores_dict.items():
        # mean_sqared_errors.append(scores["Mean Squared Error"])
        negative_log_predictive_density.append(scores["Negative Log Predictive Density"])
        # mean_standardized_log_loss.append(scores["Mean Standardized Log Loss"])
        # quantile_coverage_error.append(scores["Quantile Coverage Error"])
        root_mean_squared_error.append(scores["Root Mean Squared Error"])
        model_names.append(model_name)
    # plot all the scores as independent plots
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    print(f"model names: {model_names}")
    print(f"root mean_sqared_errors: {root_mean_squared_error}")
    print(f"negative_log_predictive_density: {negative_log_predictive_density}")
    axs[0].plot(model_names, root_mean_squared_error)
    axs[0].set_title("Root Mean Squared Error")
    axs[0].set_xticks(model_names)
    axs[1].plot(model_names, negative_log_predictive_density)
    axs[1].set_title("Negative Log Predictive Density")
    axs[1].set_xticks(model_names)
    # axs[2].plot(model_names, mean_standardized_log_loss)
    # axs[2].set_title("Mean Standardized Log Loss")
    # axs[2].set_xticks(model_names)
    # axs[3].plot(model_names, quantile_coverage_error)
    # axs[3].set_title("Quantile Coverage Error")
    # axs[3].set_xticks(model_names)
    # axs[4].plot(model_names, root_mean_squared_error)
    # axs[4].set_title("Root Mean Squared Error")
    # axs[4].set_xticks(model_names)
    # beautify the x-labels
    plt.setp(axs, xticks=model_names, xticklabels=model_names)
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
    ######################################
    # Defining paths
    #####################################
    print(os.getcwd())
    root_path = os.path.join(os.getcwd(), "kernel_composition_KDD","code","experiments_new", "synt_data_interp3")
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
    fixed_kernel_combination = ((GPy.kern.RBF(1,0.7,0.15)*GPy.kern.Matern32(1,0.6,0.03))+GPy.kern.Linear(1,0.09)) * GPy.kern.RBF(1,0.3,0.1)
    _ = np.random.uniform(-1,1,(1,1))
    fixed_model = GPy.models.GPRegression(_, _, fixed_kernel_combination)
    # make a GPy model using this kernel
    X_train = np.random.uniform(-10, 10, (100, 1))
    Y_train = fixed_model.posterior_samples_f(X_train, size=1).reshape(-1,1)
    X_test = np.random.uniform(-10, 10, (20, 1))
    # get the output of the model
    Y_test = fixed_model.posterior_samples_f(X_test,size=1).reshape(-1,1)
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
                # num_epochs = specs['num_epochs']
                # lr = specs['lr']
                stopping = specs['stopping']
                log_file_name = os.path.join(exp_path, f"heuristic_log.txt")
                metrics_log_file_name = os.path.join(exp_path, f"heuristic_metrics.txt")
                for degree_val, combination_list in zip(degrees_, combination_lists):
                    base_kernels_map = covar_kernels(kernel_list = kernels_yaml).get_base_kernels()
                    print(f"Required Degree : {degree_val}")
                    print(f"Combination Used : {combination_list}")
                    kc1 = KernelCombinations(degree_val, base_kernels_map, X_train, Y_train, X_test, Y_test,
                                            ckpt_path=ckpt_path, reinitialize=ckpt, num_initializations=5,stopping=stopping)
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
                                            progressive_model_params, degree=degree_val)
                        print(
                            f"Performing Inference for {degree_val}: {progressive_model_params.combination_name}")
                        inference.plot_fit_gpy(figure_save_path=exp_path)
                        scores_dict = inference.tabulated_scores_gpy(metrics_log_file=metrics_log_file)
                        scores_visualizer[progressive_model_params.combination_name] = scores_dict
                    # Plotting Scores
                    plot_scores(exp_path,scores_visualizer)
                    end = time.time()
                    print(f"Time taken for experiment {id} : {end-start}",file=metrics_log_file)
                    metrics_log_file.close()
                            