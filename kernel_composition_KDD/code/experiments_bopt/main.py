import GPy
import GPyOpt
import numpy as np
import os
######################
# DATA LOADING
######################
fixed_kernel_combination = GPy.kern.Matern32(input_dim=1) * \
             GPy.kern.StdPeriodic(input_dim=1, period=4) + \
             GPy.kern.RBF(input_dim=1, variance=0.9, lengthscale=3.9) + \
             GPy.kern.Matern32(input_dim=1, variance=0.3, lengthscale=1.2)
_ = np.random.uniform(-1,1,(1,1))
fixed_model = GPy.models.GPRegression(_, _, fixed_kernel_combination)
# make a GPy model using this kernel
X_train = np.random.uniform(-10, 10, (100, 1))
Y_train = fixed_model.posterior_samples_f(X_train, size=1).reshape(-1,1)
X_test = np.random.uniform(-5.5,9,(20, 1))
# Get the output of the model
Y_test = fixed_model.posterior_samples_f(X_test,size=1).reshape(-1,1)
#####################################################################################
def f(x):
    x = x.flatten()
    kernel = GPy.kern.Matern32(input_dim=1,variance=x[0],lengthscale=x[1]) * \
             GPy.kern.StdPeriodic(input_dim=1, period=x[2],variance = x[3],lengthscale=x[4]) + \
             GPy.kern.RBF(input_dim=1, variance=x[5], lengthscale=x[6]) + \
             GPy.kern.Matern32(input_dim=1, variance=x[7], lengthscale=x[8])
    model = GPy.models.GPRegression(X_train, Y_train, kernel)
    return -model.log_likelihood()

# SEARCH CONSTRAINS
bounds = [{'name': 'matern_variance', 'type': 'continuous', 'domain': (0.1, 2.0)},
        {'name': 'matern_lengthscale', 'type': 'continuous', 'domain': (0.1, 5.0)},
        {'name': 'period', 'type': 'continuous', 'domain': (0.1, 10.0)},
        {'name': 'periodic_variance', 'type': 'continuous', 'domain': (0.1, 2.0)},
        {'name': 'periodic_lengthscale', 'type': 'continuous', 'domain': (0.1, 5.0)},
          {'name': 'rbf_variance', 'type': 'continuous', 'domain': (0.1, 2.0)},
          {'name': 'rbf_lengthscale', 'type': 'continuous', 'domain': (0.1, 5.0)},
          {'name': 'matern_variance', 'type': 'continuous', 'domain': (0.1, 2.0)},
          {'name': 'matern_lengthscale', 'type': 'continuous', 'domain': (0.1, 5.0)}]
# INITIALIAZATIONS
X_init = np.random.rand(9, 9)

# BAYESIAN OPTIMIZATION
############################################
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, X=X_init)
optimizer.run_optimization(max_iter=10)
# optimizer.plot_acquisition()
# optimizer.plot_convergence()
# optimizer.plot_convergence_normalized()

x_opt = optimizer.x_opt
period_opt = x_opt[0]
rbf_variance_opt = x_opt[1]
rbf_lengthscale_opt = x_opt[2]
matern_variance_opt = x_opt[3]
matern_lengthscale_opt = x_opt[4]

# FINAL OPTIMAL KERNEL
kernel_opt = GPy.kern.Matern32(input_dim=1) * \
             GPy.kern.StdPeriodic(input_dim=1, period=period_opt) + \
             GPy.kern.RBF(input_dim=1, variance=rbf_variance_opt, lengthscale=rbf_lengthscale_opt) + \
             GPy.kern.Matern32(input_dim=1, variance=matern_variance_opt, lengthscale=matern_lengthscale_opt)
# TESTING THE OPTIMAL KERNEL
print(f"OPTIMIZATION USING GPyOpt")
model_opt = GPy.models.GPRegression(X_train, Y_train, kernel_opt)
for param_name,param_value in zip(model_opt.parameter_names(),model_opt.param_array):
            print(f"{param_name} : {param_value}")
# make predictions
Y_pred, Y_var = model_opt.predict(X_test)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_train, Y_train, c='k', marker='x', s=50, label='Training data')
# sort the data for the plot
sort_idx = X_test.flatten().argsort()
X_test_plot = X_test[sort_idx]
Y_test_plot = Y_test[sort_idx]
Y_pred_plot = Y_pred[sort_idx]
Y_var_plot = Y_var[sort_idx]
plt.plot(X_test_plot, Y_test_plot, 'r', label='True function')
plt.plot(X_test_plot, Y_pred_plot, 'b', label='Predicted function')
plt.fill_between(X_test_plot.flatten(), (Y_pred_plot - 2 * np.sqrt(Y_var_plot)).flatten(), (Y_pred_plot + 2 * np.sqrt(Y_var_plot)).flatten(), alpha=0.5, label='95% confidence interval')
plt.legend()
root_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments_bopt'
file_name = os.path.join(root_path,'results_bopt.png')
plt.savefig(file_name)

#################
# OPTIMIZING USING GPy
#################
# Define the kernel
kernel = GPy.kern.Matern32(input_dim=1) * \
            GPy.kern.StdPeriodic(input_dim=1) + \
            GPy.kern.RBF(input_dim=1) + \
            GPy.kern.Matern32(input_dim=1)
# Define the model
model_gpy_opt = GPy.models.GPRegression(X_train, Y_train, kernel)
print(model_gpy_opt.kern)
model_gpy_opt.optimize_restarts(num_restarts=10)
print(f"OPTIMIZATION USING GPy")
for param_name,param_value in zip(model_gpy_opt.parameter_names(),model_gpy_opt.param_array):
            print(f"{param_name} : {param_value}")
Y_pred, Y_var = model_gpy_opt.predict(X_test)
plt.figure()
plt.scatter(X_train, Y_train, c='k', marker='x', s=50, label='Training data')
# sort the data for the plot
sort_idx = X_test.flatten().argsort()
X_test_plot = X_test[sort_idx]
Y_test_plot = Y_test[sort_idx]
Y_pred_plot = Y_pred[sort_idx]
Y_var_plot = Y_var[sort_idx]
plt.plot(X_test_plot, Y_test_plot, 'r', label='True function')
plt.plot(X_test_plot, Y_pred_plot, 'b', label='Predicted function')
plt.fill_between(X_test_plot.flatten(), (Y_pred_plot - 2 * np.sqrt(Y_var_plot)).flatten(), (Y_pred_plot + 2 * np.sqrt(Y_var_plot)).flatten(), alpha=0.5, label='95% confidence interval')
plt.legend()
root_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/experiments_bopt'
file_name = os.path.join(root_path,'results_gpy.png')
plt.savefig(file_name)


