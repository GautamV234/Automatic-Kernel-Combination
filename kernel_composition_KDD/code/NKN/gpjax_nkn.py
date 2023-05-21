# %pip install gpjax
from pprint import PrettyPrinter
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit
from jax.config import config
from jaxutils import Dataset
import jaxkern as jk
import gpjax as gpx
import gpjax as gpx
from jaxkern import DenseKernelComputation
from jaxkern.base import AbstractKernel
from jaxkern.computations import AbstractKernelComputation
import typing as tp
pp = PrettyPrinter(indent=4)
config.update("jax_enable_x64", True)
key = jr.PRNGKey(123)


n = 100
noise = 0.3

x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise

D = Dataset(X=x, y=y)

xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)
print(x.shape,y.shape,xtest.shape,ytest.shape)


class DeepKernelFunction(AbstractKernel):
    def __init__(
        self,
        nn_params:dict,
        hidden_params:dict,
        base_kernel: AbstractKernel,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: tp.Optional[tp.List[int]] = None,
    ) -> None:
        super().__init__(
            compute_engine, active_dims, True, False, "Deep Kernel"
        )
        self.base_kernel = base_kernel
        self.nn_params = nn_params
        self.hidden_params = hidden_params

    def __call__(
        self,
        params,
        x,
        y,
    ):
        return self.base_kernel(params, x, y)

    def initialise(
        self, key: jr.KeyArray
    ) -> None:
        base_kernel_params = self.base_kernel.init_params(key)
        # print(f"base_kernel_params:")
        base_kernel_params = {"base_kernel":base_kernel_params}
        # pp.pprint(base_kernel_params)
        # print(f"nn params:\n {self.nn_params}")
        print(f"hidden_params :")
        pp.pprint(self.hidden_params)
        self._params = {**self.nn_params,**self.hidden_params, **base_kernel_params}

    def init_params(self, key: jr.KeyArray) -> tp.Dict:
        return self._params

    # This is depreciated. Can be removed once JaxKern is updated.
    def _initialise_params(self, key: jr.KeyArray) -> tp.Dict:
        return self.init_params(key)
from pprint import PrettyPrinter
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax as ox
from jax import jit
from jax.config import config
from jaxutils import Dataset
import jaxkern as jk
import gpjax as gpx
config.update("jax_enable_x64", True)
pp = PrettyPrinter(indent=4)
key = jr.PRNGKey(123)

n = 100
noise = 0.3

x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).sort().reshape(-1, 1)
f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise

D = Dataset(X=x, y=y)

xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = f(xtest)

# Primitive Kernels
pk = {'matern32': jk.Matern32(), 'matern52': jk.Matern52(), 'rbf': jk.RBF()}

# Building Module 1
def new_kernel_creator(bk:list,weights:"list[int]"):
    new_kernel = None
    for i in range(len(bk)):
      if new_kernel is None:
          # add the kernel bk[i] to the new_kernel weights[i] time since multiplying by an integer is not allowed
          if weights[i]!=0:
            for _ in range(weights[i]):
                # print(f"Adding Kernel : {list(pk.keys())[i] }")
                if new_kernel is None:
                    new_kernel = bk[i]
                else:
                    new_kernel += bk[i]
      else:
          # add the kernel bk[i] to the new_kernel weights[i] time
          if weights[i]!=0:
            # print(f"Adding Kernel : {list(pk.keys())[i]}")
            for _ in range(weights[i]):
                new_kernel += bk[i]
    return new_kernel
    
# Linear Layer
def linear_layer(W, bk:list):
    # bk = list(bk.items())
    # output_kernels
    ok = []
    assert W.shape[0] == len(bk)
    # get one column at a time
    for j in range(W.shape[1]):
        col = W[:,j]
        # convert all the weights to int
        col = [int(i) for i in col]
        new_kernel = new_kernel_creator(bk, list(col))
        ok.append(new_kernel)
    return ok
# make a random weight matrix of shape (4,6) between 0 to 10
W = jr.uniform(key, shape=(3,6), minval=0, maxval=5)
base = list(pk.values())
base_kernel_params = {}
for i,base_kernel in enumerate(base, start=1):
    base_prior = gpx.Prior(kernel = base_kernel)
    parameter_state = gpx.initialise(base_prior, key)
    base_kernel_params[f"kernel_prim_{i}"] = parameter_state.params
hidden = linear_layer(W, list(pk.values()))
W2 = jr.uniform(key, shape=(6,1), minval=0, maxval=3)
output = linear_layer(W2, hidden)
nn_params = {'linear1': W, 'linear2': W2}
hidden_kernel_params = {}
for i,hidden_kernel in enumerate(hidden, start=1):
    hidden_prior = gpx.Prior(kernel = hidden_kernel)
    parameter_state = gpx.initialise(hidden_prior, key)
    hidden_kernel_params[f"kernel_hidden_{i}"] = parameter_state.params
kernel_params = {**base_kernel_params,**hidden_kernel_params}
kernel = DeepKernelFunction(base_kernel=output[0],nn_params = nn_params,hidden_params = kernel_params)
kernel.initialise(key)
prior = gpx.Prior(kernel = kernel)

likelihood = gpx.Gaussian(num_datapoints=D.n)
# likelihood
posterior = prior * likelihood
# posterior_parameter_state.params
parameter_state = gpx.initialise(posterior, key)
# pp.pprint(parameter_state.params)
negative_mll = jit(posterior.marginal_log_likelihood(D, negative=True))
negative_mll(parameter_state.params) # ---> This is the error
pp.pprint(parameter_state.params)