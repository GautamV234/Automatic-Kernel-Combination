############################################
# Imports
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
import copy
import gpytorch
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from gpytorch.metrics import  mean_absolute_error,mean_squared_error,negative_log_predictive_density,mean_standardized_log_loss,quantile_coverage_error
############################################



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



# class ExactGPModel_MultiOutput(gpytorch.models.ExactGP):
    # """A GPyTorch ExactGP model for solving regression task.
    # Args:
    #     train_x - Tensor: Training data \n
    #     train_y - Tensor: Training labels \n
    #     kernel - GPyTorch Kernel: The GpyTorch kernel used as a cov matrix while making the model \n
    #     likelihood - GPyTorch likelihood (Usually Marginal Likelihood) corresponding to the model
    # """
    # def __init__(self, train_x, train_y,kernel,likelihood):
    #   super(ExactGPModel_MultiOutput, self).__init__(train_x, train_y, likelihood)
    #   self.mean_module = gpytorch.means.ConstantMean()
    #   self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    # def forward(self, x):
    #   mean_x = self.mean_module(x)
    #   covar_x = self.covar_module(x)
    #   ret_val =  gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    #   return ret_val

# Use data classes
class Model_characteristics():
    """
    Stores all the characterstics of a Model that we have trained
    Args:
          model - An EcactGP model \n
          likelihood - GPyTorch likelihood (Usually Marginal Likelihood) corresponding to the model
          combination_name - Str: Name of the kernel combination that was used as a cov matrix while making the model
          combination - GPyTorch Kernel : The GpyTorch kernel used as a cov matrix while making the model 
    """

    def __init__(self, model, likelihood, combination_name, combination):
        self.model = model
        # pprint.pprint(f"self.model = {self.model}")
        # self.model.load_state_dict(copy.deepcopy(model.state_dict()))
        self.likelihood = likelihood
        # self.likelihood.load_state_dict(copy.deepcopy(likelihood.state_dict()))
        self.combination_name = combination_name
        self.combination = combination


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
        scores_val:Scores: The scores object that contains the scores of the model during validation
    """

    def __init__(self, x_train, y_train, x_test, y_test, model_characteristics: Model_characteristics, scores_train: Scores, scores_val: Scores):
        self.model_characteristics = model_characteristics
        self.scores_train = scores_train
        self.scores_val = scores_val
        self.X_train = x_train
        self.Y_train = y_train
        self.X_test = x_test
        self.Y_test = y_test
        model = self.model_characteristics.model
        likelihood = self.model_characteristics.likelihood
        model.eval()
        likelihood.eval()
        self.ppd_test = likelihood(model(self.X_test))
        self.scores_test = None

    def plot_fit_gpytorch(self):
        """
        Plots the fit of the model over the test data

        Args:
            ppd_val - posterior predictive distribution
        """
        ppd = self.ppd_test
        with torch.no_grad(),gpytorch.settings.fast_pred_var():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            # Get upper and lower confidence bounds
            lower, upper = ppd.confidence_region()
            # Plot training data as black stars
            ax.plot(self.X_train.numpy(), self.Y_train.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(self.X_test.numpy(), ppd.mean.numpy(), 'b')
            # Plot the actual test data as red line
            ax.plot(self.X_test.numpy(), self.Y_test.numpy(), 'r')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(self.X_test.numpy(), lower.numpy(),
                            upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Actual','Confidence'])
            ax.set_title(f"{self.model_characteristics.combination_name}")
            # plt.show()

    def tabulate_scores(self,data = "train"):
        """
        Tabulates the scores of the model
        
        data : str: "train" or "val" or "test"
        """
        assert data in ["train","val","test"], "data should be either train or val or test"
        if data == "train":
            scores = self.scores_train
        elif data == "val":
            scores = self.scores_val
        else:
            assert self.scores_test is not None, "Test scores are not available, first run the 'compute_test_predictions' method"
            scores = self.scores_test
        scores_list = [scores.mean_squared_error, scores.negative_log_predictive_density,
                       scores.mean_standardized_log_loss, scores.quantile_coverage_error, scores.root_mean_squared_error]
        scores_list = [round(score, 3) for score in scores_list]
        scores_table = [["Mean Squared Error", scores.mean_squared_error], ["Negative Log Predictive Density", scores.negative_log_predictive_density],
                        ["Mean Standardized Log Loss", scores.mean_standardized_log_loss], [
            "Quantile Coverage Error", scores.quantile_coverage_error],
            ["Root Mean Squared Error", scores.root_mean_squared_error]]
        print(tabulate(scores_table, headers=[
              f"Score ({data})", "Value"], tablefmt="fancy_grid"))
    
    def compute_test_predictions(self):
        """
        Computes and returns the scores of the model on the test data
        """
        with torch.no_grad(),gpytorch.settings.fast_pred_var():
            self.scores_test = Scores(self.model_characteristics.combination_name,self.Y_test,self.ppd_test)
            self.scores_test.compute_scores()

class Best_model_params():
    """
    Stores the following parameters of the best model:
    rmse - Float: The rmse value of the best model \n
    model - GpyTorch model \n
    likelihood - GPyTorch likelihood (Usually Marginal Likelihood) corresponding to the model
    combination_name - Str: Name of the kernel combination that was used as a cov matrix while making the model
    combination - GPyTorch Kernel : The GpyTorch kernel used as a cov matrix while making the model
    """
    def __init__(self, rmse, model, likelihood, combination_name, combination):
        self.rmse = rmse
        self.model = model
        self.likelihood = likelihood
        self.combination_name = combination_name
        self.combination = combination

class KernelCombinations():
    def __init__(self, degree, kernel_map: dict, X_train, Y_train, X_test, Y_test, X_val, Y_val):
        self.degree = degree
        self.kernels_map = kernel_map
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val
        self.best_model_params = None
        self.use_validation = False
        self.rmse_tracker = None
        self.model_tracker = None
        self.tabulated_rmse = None
        self.best_kernel_name = None
        self.kernel_encoding = {
            (idx+1): k for idx, k in enumerate(self.kernels_map.keys())}
        # print(f"self.kernel_encoding ->  {self.kernel_encoding}")
        self.idx = len(self.kernel_encoding)    

    def get_all_kernel_combinations(self, all_kernels, all_kernel_names, all_signs):
        """Creating all possible kernel combinations from the given kernel list"""
        all_kernel_combinations = []
        all_kernel_combinations_names = []
        for i in range(len(all_signs)):
            term = None
            name = None
            for num, (kernel, kernel_name) in enumerate(zip(all_kernels, all_kernel_names)):
                if (num == 0):
                    term = kernel
                    name = f'{kernel_name}'
                else:
                    term = term + \
                        kernel if all_signs[i][num-1] == 0 else term * kernel
                    name = f'{name} + {kernel_name}' if all_signs[i][num -
                                                                     1] == 0 else f'{name} * {kernel_name}'
            all_kernel_combinations.append(term)
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
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        training_iter = 50
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(self.X_train)
            loss = -mll(output, self.Y_train)
            loss.backward()
            if (log):
                print('Iter %d/%d - Loss: %.3f' %
                      (i + 1, training_iter, loss.item()))
            optimizer.step()

    def make_predictions(self, model, likelihood, X):
        """Returns the posterior predictive distribution of the model"""
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X))
            return observed_pred

    def fit_heuristic_kernel_combinations(self, plot_fit=False, heuristic_kernel: dict = None, specific_degree=None):
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
        # if (self.X_train.shape[1] > 1):
        # plot_fit = False
        if (specific_degree):
            res_degree = specific_degree
        else:
            res_degree = self.degree
        print(
            f"Starting Heuristic Kernel Combinations for degree: {res_degree}")
        print("********************************************")
        self.kernels_map_copy = self.kernels_map.copy()
        # adding the heuristic combination to the kernel encodings
        self.kernel_encoding[self.idx+1] = heuristic_kernel['combination_name']
        # incrementing the heuristic idx
        self.heuristic_idx = self.idx+1
        # adding the heuristic kernel to the kernels_map
        self.kernels_map[(heuristic_kernel['combination_name']).replace(
            " ", "")] = heuristic_kernel['combination']
        # Getting the heuristic kernels
        self.heuristic_kernels = list(
            re.split(' + | * ', heuristic_kernel['combination_name']))
        self.heuristic_kernels = list(
            filter(('+').__ne__, self.heuristic_kernels))
        self.heuristic_kernels = list(
            filter(('*').__ne__, self.heuristic_kernels))
        # getting the degree of the heuristic combination and ensuring the required degree is greater than the heuristic degree
        self.heuristic_kernel_degree = len(self.heuristic_kernels)
        assert res_degree > self.heuristic_kernel_degree, f"Degree must be greater than Given Heuristic Kenrel Degree, {res_degree} < {self.heuristic_kernel_degree}"
        self.inp = "".join(str(key) for key in self.kernel_encoding.keys())
        self.rmse_tracker_train = {}
        self.rmse_tracker_val = {}
        self.scores_tracker_train = {}
        self.scores_tracker_val = {}
        self.model_tracker = {}
        self.best_model_params = None
        for degree in range(1, res_degree+1):
            # getting all combinations of the kernels in form of their encodings for a certain degree
            iter = itertools.combinations_with_replacement(self.inp, degree)
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
                    decoded_kernel_list = [self.kernels_map[(kernel_name).replace(
                        " ", "")] for kernel_name in decoded_kernel_names_list]
                    ##########################
                    # Generating all possible  UNIQUE combinations out of these kernels
                    all_signs = self.generate_n_signs(degree-1)
                    all_kernel_combinations, all_kernel_combination_names = self.get_all_kernel_combinations(
                        decoded_kernel_list, decoded_kernel_names_list, all_signs)
                    unique_kernel_combinations, unique_kernel_combination_names = self.get_unique_kernel_combinations(
                        self.kernels_map_copy, all_kernel_combinations, all_kernel_combination_names)
                    ##########################
                    ##############
                    # Training the selected Models and corresponding likehlihoods
                    ##############
                    model_characteristics = []
                    model_characteristics: list[Model_characteristics]
                    for kernel_comb, kernel_comb_name in zip(unique_kernel_combinations, unique_kernel_combination_names):
                        likelihood = gpytorch.likelihoods.GaussianLikelihood()
                        m = ExactGPModel(
                            self.X_train, self.Y_train, kernel_comb, likelihood)
                        try:
                            self.train_model(m, likelihood=likelihood)
                        except:
                            raise Exception("Model Training Failed")
                        new_model_characteristics = Model_characteristics(
                            m, likelihood, kernel_comb_name, kernel_comb)
                        model_characteristics.append(new_model_characteristics)
                        # m = None
                        # likelihood = None
                    ######################################################
                    # Making predictions using the trained models
                    # Checking the predicted model over train and val sets
                    ######################################################
                    for new_model_characteristics in model_characteristics:
                        m = new_model_characteristics.model
                        likelihood = new_model_characteristics.likelihood
                        kernel_comb_name = new_model_characteristics.combination_name
                        kernel_comb = new_model_characteristics.combination
                        # Making predictions
                        ppd_train = self.make_predictions(
                            model=m, likelihood=likelihood, X=self.X_train)
                        ppd_val = self.make_predictions(
                            model=m, likelihood=likelihood, X=self.X_val)
                        # Computing Scores
                        scores_train = Scores(
                            kernel_comb_name, self.Y_train, ppd_train)
                        scores_train.compute_scores()
                        scores_val = Scores(
                            kernel_comb_name, self.Y_val, ppd_val)
                        scores_val.compute_scores()
                        # Storing the scores
                        self.scores_tracker_train[kernel_comb_name] = scores_train
                        self.scores_tracker_val[kernel_comb_name] = scores_val
                        self.model_tracker[kernel_comb_name] = new_model_characteristics
                        # if (plot_fit):
                        # self.plot_fit_gpytorch(ppd_val)
                        if self.best_model_params is not None:
                            if (scores_val.root_mean_squared_error < self.best_model_params.rmse):
                                self.best_model_params.rmse = scores_val.root_mean_squared_error
                                self.best_model_params.model = m
                                self.best_model_params.likelihood = likelihood
                                self.best_model_params.combination_name = kernel_comb_name
                        else:
                            self.best_model_params = Best_model_params(
                                scores_val.root_mean_squared_error, m, likelihood, kernel_comb_name, kernel_comb)
        total_time = time.time() - start
        # print(f"Total Time of Execution: {total_time}")
        return self.scores_tracker_train, self.scores_tracker_val, self.best_model_params, self.model_tracker

    def fit_kernel_combinations(self, plot_fit=False, specific_degree=None):
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
        # if (self.X_train.shape[1] > 1):
        # plot_fit = False
        if (specific_degree):
            res_degree = specific_degree
        else:
            res_degree = self.degree
        print(f"Fitting Grid Kernel compositions for Degree {res_degree}")
        print("**************************************************")
        self.inp = "".join(str(key) for key in self.kernel_encoding.keys())
        iter = itertools.combinations_with_replacement(self.inp, res_degree)
        self.scores_tracker_train = {}
        self.scores_tracker_val = {}
        self.model_tracker = {}
        for idx, kernel_idx_list in enumerate(iter):
            kernel_idx_list = list(kernel_idx_list)
            decoded_kernel_names_list = [self.kernel_encoding[int(
                kernel_idx)] for kernel_idx in kernel_idx_list]
            kernel_list = [self.kernels_map[kernel_name]
                           for kernel_name in decoded_kernel_names_list]
            all_signs = self.generate_n_signs(res_degree-1)
            all_kernel_combinations, all_kernel_combination_names = self.get_all_kernel_combinations(
                kernel_list, decoded_kernel_names_list, all_signs)
            unique_kernel_combinations, unique_kernel_combination_names = self.get_unique_kernel_combinations(
                self.kernels_map, all_kernel_combinations, all_kernel_combination_names)
            model_characteristics = []
            for kernel_comb, kernel_comb_name in zip(unique_kernel_combinations, unique_kernel_combination_names):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                m = ExactGPModel(self.X_train, self.Y_train,
                                 kernel_comb, likelihood)
                ##############
                # Training the selected Models
                ##############
                try:
                    self.train_model(m, likelihood=likelihood)
                except:
                    raise Exception("Model Training Failed")
                new_model_characteristics = Model_characteristics(
                    m, likelihood, kernel_comb_name, kernel_comb)
                model_characteristics.append(new_model_characteristics)
                # m = None
                # likelihood = None
            for new_model_characteristics in model_characteristics:
                new_model_characteristics: Model_characteristics
                m = new_model_characteristics.model
                likelihood = new_model_characteristics.likelihood
                kernel_comb_name = new_model_characteristics.combination_name
                kernel_comb = new_model_characteristics.combination
                # Making predictions
                ppd_train = self.make_predictions(
                    model=m, likelihood=likelihood, X=self.X_train)
                ppd_val = self.make_predictions(
                    model=m, likelihood=likelihood, X=self.X_val)
                # Computing Scores
                scores_train = Scores(
                    kernel_comb_name, self.Y_train, ppd_train)
                scores_train.compute_scores()
                scores_val = Scores(kernel_comb_name, self.Y_val, ppd_val)
                scores_val.compute_scores()
                # Storing the scores
                self.scores_tracker_train[kernel_comb_name] = scores_train
                self.scores_tracker_val[kernel_comb_name] = scores_val
                # if (plot_fit):
                # self.plot_fit_gpytorch(ppd_val)
                self.model_tracker[kernel_comb_name] = new_model_characteristics
                if self.best_model_params is not None:
                    if (scores_val.root_mean_squared_error < self.best_model_params.rmse):
                        self.best_model_params.rmse = scores_val.root_mean_squared_error
                        self.best_model_params.model = m
                        self.best_model_params.likelihood = likelihood
                        self.best_model_params.combination_name = kernel_comb_name
                else:
                    self.best_model_params = Best_model_params(
                        scores_val.root_mean_squared_error, m, likelihood, kernel_comb_name, kernel_comb)
        total_time = time.time() - start
        # print(f"Total time of execution : {total_time}")
        # print(f"Best model chosen : { self.best_model_params['combination_name']}")
        return self.scores_tracker_train, self.scores_tracker_val, self.best_model_params, self.model_tracker

    def baselines(self, print_output=False):
        """Returns the tabulated baselines for the GP model"""
        if not self.rmse_tracker_val:
            raise ValueError(
                "Error: fit GP models first to generate baselines")
        average_rmse = 0.
        best_rmse = 1e6
        best_kernel_comb = ""
        for kernel_combination, rmse_val in self.rmse_tracker_val.items():
            average_rmse += rmse_val
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_kernel_comb = kernel_combination
        random_kernel_comb, random_rmse = random.choice(
            list(self.rmse_tracker_val.items()))
        average_rmse = average_rmse / len(self.rmse_tracker_val)
        average_kernel_comb = "-"
        best = ["Best", best_kernel_comb, best_rmse]
        self.best_kernel_name = best_kernel_comb
        random_ = ["Random", random_kernel_comb, random_rmse]
        average = ["Average", average_kernel_comb, average_rmse]
        baseline = [best, average, random_]
        head = ['Baseline', 'RMSE']
        if print_output:
            logging.info("Baseline results:")
            for row in baseline:
                logging.info("%s: %s, RMSE: %f", row[0], row[1], row[2])
        return baseline

    #### Main Func ####
    def main(self, heuristic=False, combination_list=None):
        """
        finds the best kernel of degree N 
        combination_list :  Parameter when heuristic = True
        """
        main_start_time = time.time()
        if heuristic:
            assert sum(
                combination_list) == self.degree, f"Degree does not match : {self.degree}!={sum(combination_list)}"
            # run one by one
            degree_of_combination = combination_list[0]
            print(
                f"Performing Grid Search for First Degree of combination : {degree_of_combination}")
            scores_tracker_train, scores_tracker_val, best_model_params, model_tracker = self.fit_kernel_combinations(
                plot_fit=False, specific_degree=degree_of_combination)
            scores_tracker_train: list[Scores]
            scores_tracker_val: list[Scores]
            best_model_params: Best_model_params
            model_tracker: dict[str, Model_characteristics]
            print(
                f"Best model for degree {degree_of_combination} : {best_model_params.combination_name}, rmse: {best_model_params.rmse}")
            current_degree = combination_list[0]
            for i in range(1, len(combination_list)):
                current_degree += combination_list[i]
                print(f"Degree reached : {current_degree}")
                previous_best_model_params = best_model_params
                previous_scores_tracker_train = copy.deepcopy(
                    scores_tracker_train)
                previous_scores_tracker_val = copy.deepcopy(scores_tracker_val)
                previous_model_tracker = model_tracker
                heuristic_kernel = {
                    'combination_name': best_model_params.combination_name, 'combination': best_model_params.combination}
                scores_tracker_train, scores_tracker_val, best_model_params, model_tracker = self.fit_heuristic_kernel_combinations(
                    heuristic_kernel=heuristic_kernel, specific_degree=current_degree)
                print(
                    f"Best Model for degree {current_degree}: { best_model_params.combination_name}, rmse: {best_model_params.rmse}")
                if (previous_best_model_params.rmse < best_model_params.rmse):
                    # TODO - Implement random flip
                    print("Model getting worse (over validation data), stopping process")
                    self.scores_tracker_train = previous_scores_tracker_train
                    self.scores_tracker_val = previous_scores_tracker_val
                    self.best_model_params = previous_best_model_params
                    self.model_tracker = previous_model_tracker
                    main_end_time = time.time() - main_start_time
                    print(
                        "*************************************************************")
                    print(f"Main Total Time of Execution : {main_end_time}")
                    return copy.deepcopy(previous_best_model_params)
            main_end_time = time.time() - main_start_time
            print("*************************************************************")
            print(f"Main Total Time of Execution : {main_end_time}")
            return copy.deepcopy(best_model_params)
        else:
            self.score_tracker_train, self.score_tracker_val, self.best_model_params, self.model_tracker = self.fit_kernel_combinations(
                False)
            main_end_time = time.time() - main_start_time
            print("*************************************************************")
            print(f"Main Total Time of Execution : {main_end_time}")
            return self.best_model_params