import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax

key = random.PRNGKey(0)

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """Radial Basis Function (RBF) kernel."""
    dist = jnp.sum((x1[:, None] - x2[None, :])**2, axis=-1)
    return variance * jnp.exp(-0.5 * dist / length_scale**2)

def matern32_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """Matern 3/2 kernel."""
    dist = jnp.sqrt(jnp.sum((x1[:, None] - x2[None, :])**2, axis=-1))
    factor = jnp.sqrt(3.0) * dist / length_scale
    return variance * (1.0 + factor) * jnp.exp(-factor)

def matern52_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """Matern 5/2 kernel."""
    dist = jnp.sqrt(jnp.sum((x1[:, None] - x2[None, :])**2, axis=-1))
    factor = jnp.sqrt(5.0) * dist / length_scale
    return variance * (1.0 + factor + 5.0/3.0*factor**2) * jnp.exp(-factor)

def linear_kernel(x1, x2, variance=1.0):
    """Linear kernel."""
    return variance * jnp.dot(x1, x2.T)

def polynomial_kernel(x1, x2, degree=2, variance=1.0):
    """Polynomial kernel."""
    return variance * (jnp.dot(x1, x2.T) + 1)**degree


# # Testing the kernel functions
# if __name__ == '__main__':
#     X = random.normal(key, (100, 2))
#     X2 = random.normal(key, (90, 2))
#     y = jnp.sin(X[:, 0]) + jnp.cos(X[:, 1])
#     print(rbf_kernel(X, X2).shape)
#     print(matern32_kernel(X, X2).shape)
#     print(matern52_kernel(X, X2).shape)
#     print(linear_kernel(X, X2).shape)
#     print(polynomial_kernel(X, X2).shape)

def relu(x):
    return jnp.maximum(0, x)

def neural_network_kernel(X, kernel_weights, kernel_funcs):
    # Compute the output of the neural network with kernel functions
    # X: input data (n_samples, n_features)
    # kernel_weights: weights of the kernel functions
    # kernel_funcs: list of kernel functions
    n_kernels = len(kernel_funcs)
    K = jnp.zeros((X.shape[0], X.shape[0]))
    # Compute the covariance matrix using the kernel functions and their weights
    for i in range(n_kernels):
        K += kernel_funcs[i](X, X, kernel_weights[i])
    
    return K

def loss_fn(X, y, kernel_weights, kernel_funcs):
    # Compute the negative log-likelihood loss between the predicted and true outputs
    K = neural_network_kernel(X, kernel_weights, kernel_funcs)
    
    try:
        L = jnp.linalg.cholesky(K)
    except:
        return jnp.inf
    
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
    nll = 0.5*jnp.dot(y.T, alpha) + jnp.sum(jnp.log(jnp.diag(L))) + 0.5*X.shape[0]*jnp.log(2*jnp.pi)
    return nll

# Generate some toy data
X = random.normal(key, (100, 2))
y = jnp.sin(X[:, 0]) + jnp.cos(X[:, 1])

# Initialize the kernel weights 
n_kernels = 3
kernel_funcs = [rbf_kernel, matern32_kernel, linear_kernel]
kernel_weights = [random.normal(key, ()) for _ in range(n_kernels)]

# Differentiably optimize the kernel weights using MLE
learning_rate = 1e-2
for i in range(50):
    print(f"EPOCH : {i+1}")
    loss = loss_fn(X, y, kernel_weights, kernel_funcs)
    print(f"loss : {loss}")
    grad_fn = grad(loss_fn, argnums=2)

    grad_vals = grad_fn(X, y, kernel_weights, kernel_funcs)
    for j,(elem_weight , elem_grad) in enumerate(zip(kernel_weights,grad_vals)):
        elem_weight  = elem_weight - learning_rate * elem_grad
        kernel_weights = kernel_weights[:j] + [elem_weight] + kernel_weights[j+1:]
    
    if i % 100 == 0:
        print(f"Step {i}, loss={loss}")

# Evaluate the neural network on new data
X_test = random.normal(key, (10, 2))
K_test = neural_network_kernel(X_test, kernel_weights, kernel_funcs)
y_pred = jnp.dot(K_test, jnp.linalg.solve(jnp.linalg.cholesky(neural_network_kernel(X, kernel_weights, kernel_funcs)).T, jnp.linalg.solve(jnp.linalg.cholesky(neural_network_kernel(X, kernel_weights, kernel_funcs)), y)))
print("Predictions:", y_pred)