############################################
# Imports
from gpytorch.metrics import mean_absolute_error, mean_squared_error, negative_log_predictive_density, mean_standardized_log_loss, quantile_coverage_error
from typing import List
from functools import reduce as reduce
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from tabulate import tabulate
import itertools
import random
import gpytorch
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import warnings
import os
from dataclasses import dataclass
from tqdm import tqdm
import yaml

warnings.filterwarnings('ignore')
############################################
############################################################
#################### New changes ###########################
# Using training loss for selection (Done)
# Implementing search over different features (Discuss)
# k fold cross validation while choosing the model (Almost Done)
# Random initializations and initializing the parameters (Done)
# Use data classes (Done)
# implement yaml file for the model (Done)
# Make it work on the gpu (Done)
# Epoch level save (Done)
# Provide a choice for if pruning (stopping) should happen or not (Done)
# Implement for MUTLIPLE features (Semi Done) Some errors need to be fixed
# LBFGS optimizer (Done)
############################################################


class ExactGPModel(gpytorch.models.ExactGP):
    """A GPyTorch ExactGP model for solving regression task.
    Args: 
        train_x - Tensor: Training data \n
        train_y - Tensor: Training labels \n
        kernel - GPyTorch Kernel: The GpyTorch kernel used as a cov matrix while making the model \n
        likelihood - GPyTorch likelihood (Usually Marginal Likelihood) corresponding to the model
    """

    def __init__(self, train_x, train_y, kernel, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        ret_val = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return ret_val


class Scores():
    """A class to compute and store the scores scores of a model. Computes the following scores: \n
     mean_absolute_error - Float: Mean absolute error \n
     mean_squared_error - Float: Mean squared error \n
     negative_log_predictive_density - Float: Negative log predictive density \n
     mean_standardized_log_loss - Float: Mean standardized log loss \n
     quantile_coverage_error - Float: Quantile coverage error \n
     root_mean_squared_error - Float: Root mean squared error \n
    Args:
        kernel_combination_name - Str: Name of the kernel combination that was used as a cov matrix while making the model \n 
        y_true - Tensor: Actual labels \n
        ppd - Multivariate Normal Distribution: Predictive posterior distribution \n
    """

    def __init__(self, kernel_combination_name, y_true, ppd):
        self.kernel_combination_name = kernel_combination_name
        self.y_true = y_true
        self.ppd = ppd

    def compute_scores(self):
        """computes the scores and stores them in the class"""
        self.mean_absolute_error = mean_absolute_error(
            self.ppd, self.y_true).item()
        self.mean_squared_error = mean_squared_error(
            self.ppd, self.y_true, squared=True).item()
        self.negative_log_predictive_density = negative_log_predictive_density(
            self.ppd, self.y_true).item()
        self.mean_standardized_log_loss = mean_standardized_log_loss(
            self.ppd, self.y_true).item()
        self.quantile_coverage_error = quantile_coverage_error(
            self.ppd, self.y_true, quantile=95.0).item()
        self.root_mean_squared_error = mean_squared_error(
            self.ppd, self.y_true, squared=False).item()


class Checkpoint_creator():
    """A class to create and store the checkpoints. For a particular model, we create a checkpoint for the following:
    1. The model
    2. The likelihood
    3. The kernel (covar module)

    Args:
        model - An ExactGP model \n
        likelihood - GPyTorch likelihood (Usually Marginal Likelihood) corresponding to the model \n
        kernel - GpyTorch Kernel
        ckpt_path - Str: Path to the checkpoint directory
        kernel_combination_name - Str: Name of the kernel combination that was used as a cov matrix while making the model
    """

    def __init__(self, model: ExactGPModel, likelihood, ckpt_path, kernel, kernel_combination_name):
        self.model = model
        self.likelihood = likelihood
        self.ckpt_path = ckpt_path
        self.kernel = kernel
        self.kernel_combination_name = kernel_combination_name

    def save_checkpoint(self):
        """Creates and saves checkpoint for the model"""
        model_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        kernel_state_dict = self.kernel.state_dict()
        state_dict = {'model': model_state_dict,
                      'likelihood': likelihood_state_dict, 'kernel': kernel_state_dict}
        # print(f"saving the following state_dict: \n {state_dict}")
        path = os.path.join(
            self.ckpt_path, f"{self.kernel_combination_name}.pt")
        torch.save(state_dict, path)

    @staticmethod
    def load_checkpoint(path):
        """Loads the checkpoint for the given kernel combination"""
        if not os.path.exists(path):
            raise Exception(f'Checkpoint {path} does not exist')
        state_dict = torch.load(path)
        return state_dict


@dataclass
class Best_model_params():
    """
    Stores the following parameters of the best model:
    training_loss - Float: The training_loss value of the best model \n
    model - GpyTorch model \n
    likelihood - GPyTorch likelihood (Usually Marginal Likelihood) corresponding to the model
    combination_name - Str: Name of the kernel combination that was used as a cov matrix while making the model
    combination - GPyTorch Kernel : The GpyTorch kernel used as a cov matrix while making the model
    initialization_number - Int: The initialization number of the best model
    epoch_number - Int: The epoch number of the best model
    """
    training_loss: float
    model: ExactGPModel
    likelihood: gpytorch.likelihoods.Likelihood
    combination_name: str
    combination: gpytorch.kernels.Kernel
    initialization_number: int = None
    epoch_number: int = None

    def view_params(self):
        """Prints the parameters of the best model"""
        # print(f"Training loss: {self.training_loss}")
        # print(f"Model: {self.model}")
        # print(f"Likelihood: {self.likelihood}")
        # print(f"Kernel combination name: {self.combination_name}")
        # print(f"Kernel combination: {self.combination}")
        assert self.combination == self.model.covar_module.base_kernel, "The kernel combination is not the same as the kernel used in the model"


@dataclass
class Model_characteristics():
    """
    Stores all the characterstics of a Model that we have trained
    Args:
          model - An EcactGP model \n
          likelihood - GPyTorch likelihood (Usually Marginal Likelihood) corresponding to the model

          combination_name - Str: Name of the kernel combination that was used as a cov matrix while making the model

          combination - GPyTorch Kernel : The GpyTorch kernel used as a cov matrix while making the model 
          training_loss - Float: The training loss of the model (MLL)
          initialization_number - Int: Which initialization of the model was this
          epoch_number : Int: The epoch number at which the model was trained

    """
    model: ExactGPModel
    likelihood: gpytorch.likelihoods.Likelihood
    combination_name: str
    combination: gpytorch.kernels.Kernel
    training_loss: float = None
    initialization_number: int = None
    epoch_number: int = None


class Inference():
    """This class is used to to perform the following:\n
    1. Plot the output of the model
    2. Output the losses of the model on train
    3. Output the losses of the model on test

    Args:
        X_train - Tensor: Training data \n
        y_train - Tensor: Training labels \n
        X_test - Tensor: Test data \n
        y_test - Tensor: Test labels \n
        model_characteristics:ModelCharacteristics: The model characteristics object
        that contains the characteristics of the model
        scores_train:Scores: The scores object that contains the scores of the model during training
    """

    def __init__(self, x_train, y_train, x_test, y_test, best_model_params: Best_model_params, scores_train: Scores = None, degree=1, device='cpu'):
        self.best_model_params = best_model_params
        self.scores_train = scores_train
        self.X_train = x_train
        self.Y_train = y_train
        self.X_test = x_test
        self.Y_test = y_test
        self.model = self.best_model_params.model
        self.likelihood = self.best_model_params.likelihood
        self.degree = degree
        self.device = device
        self.model.eval()
        self.likelihood.eval()
        print(f"self.X_test is on device {self.X_test.device}")
        print(f"self.model is on device {self.model.covar_module.base_kernel.device}")
        print(f"self.likelihood is on device {self.model.likelihood.device}")
        self.ppd_test = self.likelihood(self.model(self.X_test))
        self.model.train()
        self.likelihood.train()
        self.scores_test = None

    def plot_fit_gpytorch(self, figure_save_path):
        """
        Plots the fit of the model over the test data

        Args:
            ppd_val - posterior predictive distribution
        """
        ppd = self.ppd_test
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Initialize plot
            fig = plt.figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            # Get upper and lower confidence bounds
            lower, upper = ppd.confidence_region()
            # Plot training data as black stars
            X_train_plot = self.X_train.cpu().numpy()
            Y_train_plot = self.Y_train.cpu().numpy()
            X_test_plot = self.X_test.cpu().numpy()
            Y_test_plot = self.Y_test.cpu().numpy()
            # Get the indices that would sort the X_test_plot array
            sort_indices = np.argsort(X_test_plot)
            # Use the sort indices to sort the X_test_plot and Y_test_plot arrays
            X_test_plot = X_test_plot[sort_indices]
            Y_test_plot = Y_test_plot[sort_indices]
            lower_plot = lower.cpu().numpy()
            upper_plot = upper.cpu().numpy()
            ppd_mean_plot = ppd.mean.cpu().numpy()
            ppd_mean_plot = ppd_mean_plot[sort_indices]
            lower_plot = lower_plot[sort_indices]
            upper_plot = upper_plot[sort_indices]
            ax.plot(X_train_plot, Y_train_plot, 'k*')
            # Plot predictive means as blue line
            ax.plot(X_test_plot, ppd_mean_plot, 'b')
            # Plot the actual test data as red line
            ax.plot(X_test_plot, Y_test_plot, 'r')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(X_test_plot, lower_plot, upper_plot, alpha=0.5)
            ax.legend(['Observed Data', 'Mean', 'Actual', 'Confidence'])
            title = f"{self.degree}_{self.best_model_params.combination_name}"
            ax.set_title(title)
            plt.savefig(f"{figure_save_path}/{title}.png")
            plt.close()
        self.model.train()
        self.likelihood.train()

    def tabulate_scores(self, data="train", metrics_log_file=None):
        """
        Tabulates the scores of the model

        data : str: "train" or "test"
        """
        assert data in ["train", "test"], "data should be either train or test"
        if data == "train":
            scores = self.scores_train
        else:
            assert self.scores_test is not None, "Test scores are not available, first run the 'compute_test_predictions' method"
            scores = self.scores_test
        scores_list = [scores.mean_squared_error, scores.negative_log_predictive_density,
                       scores.mean_standardized_log_loss, scores.quantile_coverage_error, scores.root_mean_squared_error]
        scores_dict = {"Mean Squared Error": scores.mean_squared_error, "Negative Log Predictive Density": scores.negative_log_predictive_density,
                       "Mean Standardized Log Loss": scores.mean_standardized_log_loss,
                       "Quantile Coverage Error": scores.quantile_coverage_error,
                       "Root Mean Squared Error": scores.root_mean_squared_error}
        scores_list = [round(score, 3) for score in scores_list]
        scores_table = [["Mean Squared Error", scores.mean_squared_error], ["Negative Log Predictive Density", scores.negative_log_predictive_density],
                        ["Mean Standardized Log Loss", scores.mean_standardized_log_loss], [
            "Quantile Coverage Error", scores.quantile_coverage_error],
            ["Root Mean Squared Error", scores.root_mean_squared_error]]
        print(f"Results for model : {self.best_model_params.combination_name}")
        print(tabulate(scores_table, headers=[
              f"Score ({data})", "Value"], tablefmt="fancy_grid"), file=metrics_log_file)
        print("", file=metrics_log_file)
        return scores_dict

    def compute_test_predictions(self,):
        """
        Computes and returns the scores of the model on the test data
        """
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.scores_test = Scores(
                self.best_model_params.combination_name, self.Y_test, self.ppd_test)
            self.scores_test.compute_scores()
        self.model.train()
        self.likelihood.train()


class KernelCombinations():
    def __init__(self, degree, kernel_map: dict, X_train, Y_train, X_test, Y_test, X_val=None, Y_val=None, **kwargs):
        self.degree = degree
        self.base_kernels_map = {k: v for k, v in kernel_map.items()}
        # base kernels must remain intact
        self.additive_kernels_map = {k: v for k, v in kernel_map.items()}
        # additive kernels can be modified during the process with the new kernels
        self.kernels_map = kernel_map
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val
        self.best_model_params = None
        self.rmse_tracker = None
        self.model_tracker = None
        self.tabulated_rmse = None
        self.best_kernel_name = None
        self.kernel_encoding = {
            (idx+1): k for idx, k in enumerate(self.base_kernels_map.keys())}
        self.idx = len(self.kernel_encoding)
        self.reinitialize = kwargs.get("reinitialize", False)
        self.ckpt_path = kwargs.get("ckpt_path", None)
        self.device = kwargs.get("device", "cpu")
        self.num_initializations = kwargs.get("num_initializations", 1)
        self.num_epochs = kwargs.get("num_epochs", 100)
        self.lr = kwargs.get("lr", 0.1)
        self.stopping = kwargs.get("stopping", True)

    def get_all_kernel_combinations(self, all_kernels, all_kernel_names, all_signs):
        """Creating all possible kernel combinations from the given kernel list"""
        all_kernel_combinations = []
        all_kernel_combinations_names = []
        assert len(all_kernels) == len(
            all_kernel_names), "all_kernels and all_kernel_names must be of same length"
        for i in range(len(all_signs)):
            # print(f"i: {i} (i denotes which ith combination is being formed)")
            covar_mat = None
            name = None
            for num, (kernel, kernel_name) in enumerate(zip(all_kernels, all_kernel_names)):
                path = os.path.join(self.ckpt_path, f"{kernel_name}.pt")
                # Instantiating from previous values if mentioned
                if (not self.reinitialize and os.path.exists(path)):
                    state_dict = Checkpoint_creator.load_checkpoint(path)
                    kernel_state_dict = state_dict["kernel"]
                    kernel.load_state_dict(kernel_state_dict)
                else:
                    init_params = {}
                    for param_name, param in kernel.named_parameters():
                        param_name: str
                        init_params[param_name] = torch.rand(1).to(self.device)
                    kernel.initialize(**init_params)
                if (num == 0):
                    covar_mat = kernel
                    name = f'{kernel_name}'
                else:
                    covar_mat = covar_mat + \
                        kernel if all_signs[i][num -
                                               1] == 0 else covar_mat * kernel
                    name = f'{name} + {kernel_name}' if all_signs[i][num -
                                                                     1] == 0 else f'{name} * {kernel_name}'
            all_kernel_combinations.append(covar_mat)
            all_kernel_combinations_names.append(name)
        return all_kernel_combinations, all_kernel_combinations_names

    def generate_n_signs(self, n=1):
        """Generates n different signs for n+1 kernel combination grid"""
        """ 0 denotes +, 1 denotes * """
        all_signs = []
        signs = [-1]*n
        self.recur(signs, 0, all_signs)
        return all_signs

    def recur(self, signs: List[int], idx: int, all_signs: List[List[int]]):
        if idx == len(signs):
            all_signs.append(signs)
            return
        self.recur(signs[:idx] + [0] + signs[idx+1:], idx+1, all_signs)
        self.recur(signs[:idx] + [1] + signs[idx+1:], idx+1, all_signs)

    def get_unique_kernel_combinations(self, kernels_map, all_kernel_combinations, all_kernel_combination_names, seed=2000):
        """
        Returns the filtered list of unique kernel combinations from all possible kernel combinations.
        Parameters:
        - kernels_map: The initial map of kernels that is used for creating the kernel combinations.
        - all_kernel_combination_names: Names of all kernel combinations formed with any number of chosen kernels.
        - seed: A random seed used to generate the numerized codes for the kernel names.
        """
        unique_combination_list = []
        unique_combination_names_list = []
        unique_scores = []
        # Create a map of kernel names to numerized codes
        kernel_number_map = {kernel_name: random.randint(
            0, seed) for kernel_name in kernels_map.keys()}
        # Iterate through the kernel combinations and their names
        for kernel_combination, kernel_combination_name in zip(all_kernel_combinations, all_kernel_combination_names):
            # Split the kernel combination name into its component kernel names
            kernel_combination_name_split = kernel_combination_name.split(" ")
            # Initialize a list to hold the numerized kernel names
            numerized_kernel_name_split = [0] * \
                len(kernel_combination_name_split)
            # Iterate through the kernel names in the kernel combination name
            for i, kernel_name in enumerate(kernel_combination_name_split):
                # If the element is "+" or "*", keep it as is
                if kernel_name == "+" or kernel_name == "*":
                    numerized_kernel_name_split[i] = kernel_name
                # Otherwise, replace the element with its corresponding value in kernel_number_map
                else:
                    numerized_kernel_name_split[i] = kernel_number_map[kernel_name]
            # Calculate the value of the numerized kernel combination name
            value = self.solve_term(numerized_kernel_name_split)
            # If the value has not been seen before, add the kernel combination and its name to the lists
            if value not in unique_scores:
                unique_combination_list.append(kernel_combination)
                unique_combination_names_list.append(kernel_combination_name)
                unique_scores.append(value)
        # Return the lists of unique kernel combinations and their names
        return unique_combination_list, unique_combination_names_list

    def solve_term(self, numerized_kernel_name_split: list):
        """Finds the value for a particular kernel combination for uniqueness correction"""
        i = 0
        stack = []
        value = 0
        while (i < len(numerized_kernel_name_split)):
            elem = numerized_kernel_name_split[i]
            if (elem != '+'):
                stack.append(elem)
            else:
                temp_val = 1
                j = 0
                while (j < len(stack)):
                    if (stack[j] != '*'):
                        temp_val = temp_val*stack[j]
                    else:
                        pass
                    j += 1
                value += temp_val
                stack = []
            i += 1
        # finishing the last elements
        if (len(stack) > 0):
            temp_val = 1
            j = 0
            while (j < len(stack)):
                if (stack[j] != '*'):
                    temp_val = temp_val*stack[j]
                j += 1
            value += temp_val
        return value

    def train_model(self, model, likelihood, log=False):
        """Training the GPytorch model
        Parameters:
        - model: The GPytorch model to be trained.
        - likelihood: The likelihood function to be used for training.
        Returns:
        - final loss
        - list of state dict of the model after training of every 100 epochs.
        WHATS NEW:
         Storing the model state dict every 100 epochs to understand the affect of training on the model.
        """
        epochwise_model_state_dict_saver = {}
        # epochwise_model_state_dict_saver : dict
        # key - epoch number
        # value - (training_loss,model state dict)
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=self.lr)
        training_iter = self.num_epochs
        def closure():
            optimizer.zero_grad()
            output = model(self.X_train)
            loss = -output.log_prob(self.Y_train).sum()
            loss.backward()
            return loss
        final_loss = None
        for i in tqdm(range(training_iter)):
            loss = optimizer.step(closure)
            if (log):
                print('Iter %d/%d - Loss: %.3f' %
                      (i + 1, training_iter, loss.item()))
            if(i % 100 == 0):
                epochwise_model_state_dict_saver[i] = (model.state_dict(), loss.item())
        final_loss = loss.item()
        return final_loss, epochwise_model_state_dict_saver

    def make_predictions(self, model, likelihood, X):
        """Returns the posterior predictive distribution of the model"""
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X))
            return observed_pred

    def tabulate_training_results(self, model_characteristics, file=None):
        """Tabulates the training results of the models.

        Args:
            model_characteristics: list[Model_characteristics] - list of models and their characteristics.
        """
        print("Tabulating training results...")
        results_tabulated = []
        for model_characteristic in model_characteristics:
            model_characteristic:Model_characteristics
            results_tabulated.append([model_characteristic.combination_name,
                                     model_characteristic.initialization_number, model_characteristic.epoch_number ,model_characteristic.training_loss])
        if(results_tabulated == []):
            return
        print(tabulate(results_tabulated, headers=[
              "Kernel Combination", "Initialization Number", "Epoch Number", "Training Loss"], tablefmt="github"), file=file)
        print("", file=file)

    def fit_heuristic_kernel_combinations(self, plot_fit=False, heuristic_kernel: dict = None, specific_degree=None, log_file=None):
        """Fits GP models using different kernel combinations heuristically.

        Args:
            plot_fit: If True, plots the fit.
            heuristic_kernel: Dict with the following keys and values:
                'combination_name': name of the combination (str)
                'combination': GPy Kernel
            specific_degree: If specified, only fits models of this degree.

        Returns:
            scores_tracker_train: list[Scores] - scores for training data.
            scores_tracker_val: list[Scores] - scores for validation data.
            best_model_params: Best_model_params - details of the best model\n
            model_tracker: dict -> {(str) kernel combination name : Model_characteristics class}
        """
        start = time.time()
        if (specific_degree):
            res_degree = specific_degree
        else:
            res_degree = self.degree
        print(
            f"Starting Heuristic Kernel Combinations for degree: {res_degree}")
        print("********************************************")
        # Initialize the kernels with their values from the previous iteration if reinitialize is False
        # adding the heuristic combination to the kernel encodings
        self.kernel_encoding[self.idx+1] = heuristic_kernel['combination_name']
        # incrementing the heuristic idx
        self.heuristic_idx = self.idx+1
        # adding the heuristic kernel to the kernels_map
        self.additive_kernels_map[(heuristic_kernel['combination_name']).replace(
            " ", "")] = heuristic_kernel['combination']
        # Getting the heuristic kernels
        ####################################
        self.heuristic_kernels = list(
            re.split(' + | * ', heuristic_kernel['combination_name']))
        self.heuristic_kernels = list(
            filter(('+').__ne__, self.heuristic_kernels))
        self.heuristic_kernels = list(
            filter(('*').__ne__, self.heuristic_kernels))
        ####################################
        # getting the degree of the heuristic combination and ensuring the required degree is greater than the heuristic degree
        self.heuristic_kernel_degree = len(self.heuristic_kernels)
        assert res_degree > self.heuristic_kernel_degree, f"Degree must be greater than Given Heuristic Kenrel Degree, {res_degree} < {self.heuristic_kernel_degree}"
        self.kernel_encoding_keys_iterable = [str(key) for key in self.kernel_encoding.keys()]
        self.scores_tracker_train = {}
        self.scores_tracker_val = {}
        self.model_tracker = {}
        self.best_model_params = None
        for degree in range(1, res_degree+1):
            # getting all combinations of the kernels in form of their encodings for a certain degree
            iter = itertools.combinations_with_replacement(
                self.kernel_encoding_keys_iterable, degree)
            model_characteristics = []
            model_characteristics: list[Model_characteristics]
            for idx, kernel_idx_list in enumerate(iter):
                if (str(self.heuristic_idx) not in kernel_idx_list):
                    # if our heuristic idx is not present in the kernel idx list then we skip as we want to build on top of that
                    continue
                else:
                    # Getting the degree of the combination for the suggested combination
                    count = len(
                        [elem for elem in kernel_idx_list if elem == str(self.heuristic_idx)])
                    iter_degree = degree - count + count * self.heuristic_kernel_degree
                    # ensuring the iteration degree is same as the required degree otherwise we skip on this proposed combination
                    if (iter_degree != res_degree):
                        continue
                    kernel_idx_list = list(kernel_idx_list)
                    # getting the name of the kernels from the encodings
                    decoded_kernel_names_list = [self.kernel_encoding[int(
                        kernel_idx)] for kernel_idx in kernel_idx_list]
                    # trimming to reduce uneccesary errors
                    decoded_kernel_list = [self.additive_kernels_map[(kernel_name).replace(
                        " ", "")] for kernel_name in decoded_kernel_names_list]
                    ##########################
                    # Generating all possible UNIQUE combinations out of these kernels
                    all_signs = self.generate_n_signs(degree-1)
                    all_kernel_combinations, all_kernel_combination_names = self.get_all_kernel_combinations(
                        decoded_kernel_list, decoded_kernel_names_list, all_signs)
                    unique_kernel_combinations, unique_kernel_combination_names = self.get_unique_kernel_combinations(
                        self.base_kernels_map, all_kernel_combinations, all_kernel_combination_names)
                    ##########################
                    ##############
                    # Training the selected Models and corresponding likehlihoods
                    ##############
                    for kernel_comb, kernel_comb_name in zip(unique_kernel_combinations, unique_kernel_combination_names):
                        for initialisation_num in range(1, self.num_initializations+1):
                            print(
                                f"Initialisation number: {initialisation_num}")
                            likelihood = gpytorch.likelihoods.GaussianLikelihood()
                            m = ExactGPModel(self.X_train, self.Y_train,
                                             kernel_comb, likelihood)
                            ##############
                            # Training the selected Models
                            ##############
                            m = m.to(self.device)
                            likelihood = likelihood.to(self.device)
                            print(f"Model being trained: {kernel_comb_name}")
                            try:
                                training_loss, epochwise_model_state_dict_saver = self.train_model(
                                    m, likelihood=likelihood)
                            except Exception as e:
                                print("Error in training model")
                                print(e)
                                raise Exception("Model Training Failed")
                            for key, value in epochwise_model_state_dict_saver.items():
                                new_likelihood = gpytorch.likelihoods.GaussianLikelihood()
                                new_model = ExactGPModel(self.X_train, self.Y_train,
                                                         kernel_comb, new_likelihood)
                                # value[0] is the model state dict
                                # value[1] is the training_loss
                                new_model.load_state_dict(value[0])
                                new_model_characteristics = Model_characteristics(
                                    new_model, new_likelihood, kernel_comb_name, kernel_comb, value[1], initialization_number=initialisation_num, epoch_number=key)
                                model_characteristics.append(
                                    new_model_characteristics)
                            # Adding the latest model in case the num of epochs is not a multiple of 100
                            if(self.num_epochs not in epochwise_model_state_dict_saver.keys()):
                                new_model_characteristics = Model_characteristics(
                                    m, likelihood, kernel_comb_name, kernel_comb, training_loss, initialization_number=initialisation_num, epoch_number=self.num_epochs)
                                model_characteristics.append(
                                    new_model_characteristics)
            ######################################################
            # Making predictions using the trained models
            # Checking the predicted model over train and val sets
            ######################################################
            for new_model_characteristics in model_characteristics:
                m = new_model_characteristics.model
                likelihood = new_model_characteristics.likelihood
                kernel_comb_name = new_model_characteristics.combination_name
                kernel_comb = new_model_characteristics.combination
                training_loss = new_model_characteristics.training_loss
                initialisation_num - new_model_characteristics.initialization_number
                epoch_number = new_model_characteristics.epoch_number
                self.model_tracker[kernel_comb_name] = new_model_characteristics
                if self.best_model_params is not None:
                    if (training_loss < self.best_model_params.training_loss):
                        self.best_model_params.training_loss = training_loss
                        self.best_model_params.model = m
                        self.best_model_params.likelihood = likelihood
                        self.best_model_params.combination_name = kernel_comb_name
                        self.best_model_params.combination = kernel_comb
                        self.best_model_params.initialization_number = initialisation_num
                        self.best_model_params.epoch_number = epoch_number
                else:
                    self.best_model_params = Best_model_params(
                        training_loss, m, likelihood, kernel_comb_name, kernel_comb, initialisation_num, epoch_number)
                # viewing params
                self.best_model_params.view_params()
            self.tabulate_training_results(model_characteristics, log_file)

        # Saving checkpoint
        ckpt_obj = Checkpoint_creator(self.best_model_params.model, self.best_model_params.likelihood,
                                      self.ckpt_path, self.best_model_params.combination, self.best_model_params.combination_name)
        ckpt_obj.save_checkpoint()
        total_time = time.time() - start
        return self.scores_tracker_train, self.scores_tracker_val, self.best_model_params, self.model_tracker

    def fit_kernel_combinations(self, plot_fit=False, specific_degree=None, log_file=None):
        """Fits GP models using different kernel combinations heuristically.
        WHATS NEW - 1. Added the ability to fit models for multidimensional data
        Args:
            plot_fit: If True, plots the fit.
            heuristic_kernel: Dict with the following keys and values:
                'combination_name': name of the combination (str)
                'combination': GPy Kernel
            specific_degree: If specified, only fits models of this degree.

        Returns:
            scores_tracker_train: list[Scores] - scores for training data.
            scores_tracker_val: list[Scores] - scores for validation data.
            best_model_params: Best_model_params - details of the best model\n
            model_tracker: dict -> {(str) kernel combination name : Model_characteristics class}
        """
        start = time.time()
        if (specific_degree):
            res_degree = specific_degree
        else:
            res_degree = self.degree
        print(f"Fitting Grid Kernel compositions for Degree {res_degree}")
        print("**************************************************")
        self.kernel_encoding_keys_iterable = [str(key) for key in self.kernel_encoding.keys()]
        iter = itertools.combinations_with_replacement(
            self.kernel_encoding_keys_iterable, res_degree)
        self.scores_tracker_train = {}
        self.scores_tracker_val = {}
        self.model_tracker = {}
        model_characteristics = []
        for idx, kernel_idx_list in enumerate(iter):
            kernel_idx_list = list(kernel_idx_list)
            print(f"kernel_idx_list: {kernel_idx_list}")
            decoded_kernel_names_list = [self.kernel_encoding[int(
                kernel_idx)] for kernel_idx in kernel_idx_list]
            kernel_list = [self.additive_kernels_map[kernel_name]
                           for kernel_name in decoded_kernel_names_list]
            all_signs = self.generate_n_signs(res_degree-1)
            all_kernel_combinations, all_kernel_combination_names = self.get_all_kernel_combinations(
                kernel_list, decoded_kernel_names_list, all_signs)
            unique_kernel_combinations, unique_kernel_combination_names = self.get_unique_kernel_combinations(
                self.base_kernels_map, all_kernel_combinations, all_kernel_combination_names)
            for kernel_comb, kernel_comb_name in zip(unique_kernel_combinations, unique_kernel_combination_names):
                for initialisation_num in range(1, self.num_initializations+1):
                    print(f"Initialisation number: {initialisation_num}")
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    m = ExactGPModel(self.X_train, self.Y_train,
                                     kernel_comb, likelihood)
                    ##############
                    # Training the selected Models
                    ##############
                    m = m.to(self.device)
                    likelihood = likelihood.to(self.device)
                    print(f"Model being trained: {kernel_comb_name}")
                    try:
                        training_loss, epochwise_model_state_dict_saver = self.train_model(
                            m, likelihood=likelihood)
                    except Exception as e:
                        print("Error in training model")
                        print(e)
                        raise Exception("Model Training Failed")
                    for key, value in epochwise_model_state_dict_saver.items():
                                new_likelihood = gpytorch.likelihoods.GaussianLikelihood()
                                new_model = ExactGPModel(self.X_train, self.Y_train,
                                                         kernel_comb, new_likelihood)
                                # value[0] is the model state dict
                                # value[1] is the training_loss
                                new_model.load_state_dict(value[0])
                                new_model_characteristics = Model_characteristics(
                                    new_model, new_likelihood, kernel_comb_name, kernel_comb, value[1], initialization_number=initialisation_num, epoch_number=key)
                                model_characteristics.append(
                                    new_model_characteristics)
                    # Adding the latest model in case the num of epochs is not a multiple of 100
                    if(self.num_epochs not in epochwise_model_state_dict_saver.keys()):
                        new_model_characteristics = Model_characteristics(
                            m, likelihood, kernel_comb_name, kernel_comb, training_loss, initialization_number=initialisation_num, epoch_number=self.num_epochs)
                        model_characteristics.append(
                            new_model_characteristics)
        for new_model_characteristics in model_characteristics:
            new_model_characteristics: Model_characteristics
            m = new_model_characteristics.model
            likelihood = new_model_characteristics.likelihood
            training_loss = new_model_characteristics.training_loss
            kernel_comb_name = new_model_characteristics.combination_name
            kernel_comb = new_model_characteristics.combination
            self.model_tracker[kernel_comb_name] = new_model_characteristics
            if self.best_model_params is not None:
                if (training_loss < self.best_model_params.training_loss):
                    self.best_model_params.training_loss = training_loss
                    self.best_model_params.model = m
                    self.best_model_params.likelihood = likelihood
                    self.best_model_params.combination_name = kernel_comb_name
                    self.best_model_params.combination = kernel_comb
            else:
                self.best_model_params = Best_model_params(
                    training_loss, m, likelihood, kernel_comb_name, kernel_comb)
        self.tabulate_training_results(model_characteristics, log_file)
        # viewing the params
        self.best_model_params.view_params()
        # Saving checkpoint
        ckpt_obj = Checkpoint_creator(self.best_model_params.model, self.best_model_params.likelihood,
                                      self.ckpt_path, self.best_model_params.combination, self.best_model_params.combination_name)

        path = os.path.join(
            self.ckpt_path, f"{self.best_model_params.combination_name}.pt")
        ckpt_obj.save_checkpoint()
        total_time = time.time() - start
        return self.scores_tracker_train, self.scores_tracker_val, self.best_model_params, self.model_tracker

    #### Main Func ####
    def main(self, heuristic=False, combination_list=None, log_file=None, num_epochs=100, lrate=0.1):
        """
        finds the best kernel of degree N 
        combination_list :  Parameter when heuristic = True
        """
        main_start_time = time.time()
        progressive_models = []
        if heuristic:
            assert sum(
                combination_list) == self.degree, f"Degree does not match : {self.degree}!={sum(combination_list)}"
            degree_of_combination = combination_list[0]
            print(
                f"Performing Grid Search for First Degree of combination : {degree_of_combination}")
            scores_tracker_train, scores_tracker_val, best_model_params, model_tracker = self.fit_kernel_combinations(
                plot_fit=False, specific_degree=degree_of_combination, log_file=log_file)
            scores_tracker_train: list[Scores]
            scores_tracker_val: list[Scores]
            best_model_params: Best_model_params
            model_tracker: dict[str, Model_characteristics]
            print(
                f"Best model for degree {degree_of_combination} : {best_model_params.combination_name}, training_loss: {best_model_params.training_loss}")
            progressive_models.append(best_model_params)
            current_degree = combination_list[0]
            for i in range(1, len(combination_list)):
                current_degree += combination_list[i]
                print(f"Degree reached : {current_degree}")
                previous_best_model_params = best_model_params
                previous_model_tracker = model_tracker
                heuristic_kernel = {
                    'combination_name': best_model_params.combination_name, 'combination': best_model_params.combination}
                _, _, best_model_params, model_tracker = self.fit_heuristic_kernel_combinations(
                    heuristic_kernel=heuristic_kernel, specific_degree=current_degree, log_file=log_file)
                print(
                    f"Best Model for degree {current_degree}: { best_model_params.combination_name}, training : {best_model_params.training_loss}")
                if (self.stopping and previous_best_model_params.training_loss < best_model_params.training_loss):
                    # TODO - Implement random flip
                    print(
                        "Model getting worse (over training loss), stopping process")
                    print(
                        f"Overall best model : {previous_best_model_params.combination_name} : {previous_best_model_params.training_loss}")
                    progressive_models.append(previous_best_model_params)
                    main_end_time = time.time() - main_start_time
                    print(
                        "*************************************************************")
                    print(f"Main Total Time of Execution : {main_end_time}")
                    return previous_best_model_params, progressive_models
                progressive_models.append(best_model_params)
            print(
                f"Overall best model : {best_model_params.combination_name} : {best_model_params.training_loss}")
            main_end_time = time.time() - main_start_time
            print("*************************************************************")
            print(f"Main Total Time of Execution : {main_end_time}")
            return best_model_params, progressive_models
        # else:
        #     score_tracker_train, score_tracker_val, best_model_params, model_tracker = self.fit_kernel_combinations(
        #         False)
        #     main_end_time = time.time() - main_start_time
        #     print("*************************************************************")
        #     print(f"Main Total Time of Execution : {main_end_time}")
        #     return best_model_params
