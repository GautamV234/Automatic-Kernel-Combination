import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
key = random.PRNGKey(0)

def relu(x):
    return jnp.maximum(0, x)

def linear_kernel(x, y, scale=1.0):
    # Compute the linear kernel between x and y
    return scale * jnp.dot(x, y.T)

def rbf_kernel(x, y, gamma=1.0):
    # Compute the RBF kernel between x and y
    pairwise_dists = jnp.sum(x**2, axis=-1, keepdims=True) + jnp.sum(y**2, axis=-1) - 2 * jnp.dot(x, y.T)
    return jnp.exp(-gamma * pairwise_dists)

def neural_network_kernel(X, kernel_weights):
    # Compute the output of the neural network with kernel functions
    # X: input data (n_samples, n_features)
    # kernel_weights: weights of the kernel functions
    hidden_layer = relu(jnp.dot(X, kernel_weights[0]))
    output_layer = jnp.dot(hidden_layer, kernel_weights[1])
    return output_layer

def loss_fn(X, y, kernel_weights):
    # Compute the mean squared error loss between the predicted and true outputs
    y_pred = neural_network_kernel(X, kernel_weights)
    return jnp.mean((y_pred - y)**2)

# Generate some toy data
X = random.normal(key, (100, 2))
y = jnp.sin(X[:, 0]) + jnp.cos(X[:, 1])

# Initialize the kernel weights 
kernel_weights = [
    random.normal(key, (2, 4)),  # weights for the first kernel function
    random.normal(key, (4, 1))   # weights for the second kernel function
]
# Differentiably optimize the kernel weights
learning_rate = 1e-2
for i in range(5):
    print(f"EPOCH : {i+1}")
    loss = loss_fn(X, y, kernel_weights)
    grad_fn = grad(loss_fn, argnums=2)

    grad_vals = grad_fn(X, y, kernel_weights)
    for i,(weight_, grad_) in enumerate(zip(kernel_weights, grad_vals)):
        for j,(elem_weight , elem_grad) in enumerate(zip(weight_,grad_)):
            elem_weight  = elem_weight - learning_rate * elem_grad
            kernel_weights[i] = kernel_weights[i].at[j].set(elem_weight)
         
    if i % 100 == 0:
        print(f"Step {i}, loss={loss}")

# Evaluate the neural network on new data
X_test = random.normal(key, (10, 2))
y_pred = neural_network_kernel(X_test, kernel_weights)
print("Predictions:", y_pred)
