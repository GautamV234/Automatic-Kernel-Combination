import jax
import jax.numpy as jnp
import gpjax as gpj
import jaxkern as jk
# Define the three kernels
# jk.RBF()
kernel_rbf = jk.RBF(variance=1.0, lengthscale=1.0)
kernel_matern12 = jk.Matern12(variance=1.0,lengthscale=1.0)
kernel_matern32 = jk.Matern32(variance=1.0, lengthscale=1.0)

# Define the weight matrix
weights = jax.random.normal(key=0, shape=(6, 2))

# Define the kernel function that combines the three kernels with the weights
def combined_kernel_fn(X1, X2):
    K_rbf = kernel_rbf(X1, X2)
    K_matern32 = kernel_matern32(X1, X2)
    K_combined = jnp.matmul(jnp.concatenate([K_rbf[..., None],
     K_matern32[..., None]], axis=-1), weights)
    return K_combined

# Define the Gaussian process model with the combined kernel function
gp_model = gpj.models.GPRegression(kernel_fn=combined_kernel_fn, noise_var=0.01)
