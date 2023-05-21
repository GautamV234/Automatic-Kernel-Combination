import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import gpytorch
import matplotlib.pyplot as plt
import scipy.io as sio
# import numpy as np
from sklearn.utils import shuffle

i = 0
# Defining the neural network
class KernelNetwork(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(KernelNetwork, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        # rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # matern32_kernel  = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        # lin_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        # self.rbf_kernel = rbf_kernel        
        # self.matern32_kernel = matern32_kernel
        # self.lin_kernel = lin_kernel
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        matern32_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        lin_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        period_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        rational_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        self.periodic_kernel = period_kernel.type(torch.float64)
        self.rbf_kernel = rbf_kernel.type(torch.float64)
        self.matern32_kernel = matern32_kernel.type(torch.float64)
        self.rq_kernel = rational_kernel.type(torch.float64)
        self.lin_kernel = lin_kernel.type(torch.float64)
        self.n = 5
        # self.hidden_values = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(1) * (n-1) + 1, requires_grad=True) for i in range(n)])
        self.hidden_values = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(1, dtype=torch.float64) * (self.n-1) + 1, requires_grad=True) for i in range(self.n)])
        self.mean_module = gpytorch.means.ConstantMean()
    
    def is_positive_semidefinite(matrix):
        eigvals, _ = torch.linalg.eig(matrix)
        return torch.all(eigvals[:, 0] >= 0)
        
    def forward(self, x):
        # rbf_out = self.rbf_kernel(x).evaluate()
        # matern32_out = self.matern32_kernel(x).evaluate()
        # lin_out = self.lin_kernel(x).evaluate()
        rbf_out = self.rbf_kernel(x).evaluate().type(torch.float64)
        matern32_out = self.matern32_kernel(x).evaluate().type(torch.float64)
        lin_out = self.lin_kernel(x).evaluate().type(torch.float64)
        periodic_out = self.periodic_kernel(x).evaluate().type(torch.float64)
        rq_out = self.rq_kernel(x).evaluate().type(torch.float64)
        tensors = [rbf_out, matern32_out, lin_out,periodic_out,rq_out]
        covar_x= sum([torch.log(self.hidden_values[i]+10)*tensors[i] for i in range(self.n)]) 
        mean_x = self.mean_module(x)
        is_symmetric = torch.allclose(covar_x, covar_x.T)
        print("###################################################")
        print("Is symmetric:", is_symmetric)
        # Check if the matrix is non-singular
        is_singular = torch.det(covar_x) == 0
        print("Is singular:", is_singular)
        # Check if the matrix has non-zero eigenvalues
        eigvals, eigvecs = torch.linalg.eig(covar_x)
        is_psd = torch.all(eigvals.real >= 0)
        print("Is PSD:", is_psd)
        try :
            global i
            i+=1
            ret_val =  gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            print("Non dummy iterations: ", i) 
            return ret_val
        except:
            if not is_psd or is_singular:
                if (is_singular):
                    print("Singular matrix")
                else:
                    print("Not PSD")
                lmbda = 0.1
                # I = torch.eye(100)  # Create the identity matrix of size (100, 100)
                I = torch.eye(x.shape[0], dtype=torch.float64).to(x.device)
                covar_x = covar_x + lmbda*I  # Add the diagonal matrix to A
                eigvals, eigvecs = torch.linalg.eig(covar_x)
                eigvals_real = eigvals.real.view(-1,1)
                eigvals_real_matrix = eigvals_real
                plt.imshow(eigvals_real_matrix.detach().cpu(), cmap='gray')
                plt.colorbar()
                # plt.title('Real Part of Eigenvalues')
                # global i
                # plt.savefig(f'eigvals_real_matrix_{i}.png')
                is_psd = torch.all(eigvals.real >= 0)
                print("Retry Is PSD:", is_psd)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(model, train_x, train_y, optimizer, likelihood):
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)    
    optimizer.zero_grad()
    output = model(train_x)
    # import pdb; pdb.set_trace()
    loss = -mll(output, train_y)
    loss.backward()
    # checking if gradient is flowing backwards or not
    # for i in range(len(model.hidden_values)):
        # print(f"Gradient for hidden scalar {i}: {model.hidden_values[i].grad}")
    # for i in range(len(list(model.parameters()))):
        # print(f"Gradient for parameter {i}: {list(model.parameters())[i].grad}")
    optimizer.step()
    return loss.item()

########
# Training data
########
# train_x = torch.linspace(0, 1, 100)
# train_y = torch.sin(train_x * (2 * math.pi))
# train_y += 0.1 * torch.randn_like(train_y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train_x = torch.linspace(0, 1, 100, dtype=torch.float64).to(device=device)
# train_y = torch.sin(train_x * (2 * math.pi)).type(torch.float64).to(device=device)
# train_y += 0.1 * torch.randn_like(train_y, dtype=torch.float64).to(device=device)
# test_x = torch.linspace(0.5, 1.3, 20, dtype=torch.float64).to(device=device)
# test_y = torch.sin(test_x * (2 * math.pi)).type(torch.float64).to(device=device)


def data_loader(data_path):
    data = sio.loadmat(data_path)
    X = data['X']
    y = data['y']
    X = np.array(X).reshape(-1)
    y = np.array(y).reshape(-1)
    return X, y

# Moana Loa
data_path = '/home/gautam.pv/nlim/kernel_composition_KDD/data/mauna2011.mat'
X, y = data_loader(data_path)
# get the train and test_x data by keeping the last 20% of the data as test_x data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=False)
# assume we have data X and target y
# split data into train and test_x


# n = len(X)
# split = int(n * 0.8)

# # import pdb; pdb.set_trace()
# train_x = X[:split]
# train_y = y[:split]
# test_x = X[split:]
# test_y = y[split:]


# # take last 20% of train_x as test2
# n_train = len(train_x)
# split2 = int(n_train * 0.8)

# test2 = train_x[split2:]
# test2_target = train_y[split2:]

# # use remaining 80% of train_x as new train_x
# train_x = train_x[:split2]
# train_y = train_y[:split2]

# # copy 10% of new train_x and add noise
# n_train = len(train_x)
# n_copy = int(n_train * 0.1)

# copy = train_x[:n_copy]
# copy_noisy = copy + np.random.normal(size=copy.shape)

# # add copy to test_x data
# test_x = np.concatenate((test_x, copy_noisy))
# test_y = np.concatenate((test_y, train_y[:n_copy]))


# # sort test data based on the first column
# test_x = test_x[np.argsort(test_x[:])]
# test_y = test_y[np.argsort(test_x[:])]

# shuffle train and test_x data
# train_x, train_y = shuffle(train_x, train_y)
# test_x, test_y = shuffle(test_x, test_y)
#  add the first 10% training points of test_x to train_x and dont remove them from test_x
# n_test = len(test_x)
# n_copy = int(n_test * 0.1)
# train_x = np.concatenate((train_x, test_x[:n_copy]))
# y_new = test_y[:n_copy]
# train_y = np.concatenate((train_y, y_new))

#########################
# Convert to torch tensors
#########################
complete_data_X = torch.from_numpy(X).to(torch.float64).squeeze().to(device)
complete_data_y = torch.from_numpy(y).to(torch.float64).squeeze().to(device)
train_x = torch.from_numpy(train_x).to(torch.float64).squeeze().to(device)
train_y = torch.from_numpy(train_y).to(torch.float64).squeeze().to(device)
test_x = torch.from_numpy(test_x).to(torch.float64).squeeze().to(device)
test_y = torch.from_numpy(test_y).to(torch.float64).squeeze().to(device)

# Initialize the model and likelihood
model = KernelNetwork(train_x, train_y).to(device=device)
likelihood = model.likelihood.to(device=device)

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 500
for i in range(num_epochs):
    loss = train(model, train_x, train_y, optimizer, likelihood)
    print(f"Epoch {i+1}/{num_epochs}: Loss = {loss:.3f}")
for z in range(len(model.hidden_values)):
    val = math.log(model.hidden_values[z].item()+1)
    print(f"Final hidden scalar {z}: {val}")

# f_preds = model(test_x)
# y_preds = likelihood(model(test_x))
# f_mean = f_preds.mean
# f_var = f_preds.variance
# f_covar = f_preds.covariance_matrix
# f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(complete_data_X))


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    # import pdb; pdb.set_trace()
    ax.plot(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy(), 'g*')
    # Plot predictive means as blue line
    ax.plot(complete_data_X.detach().cpu().numpy(), observed_pred.mean.detach().cpu().numpy(), 'b*')
    # Shade between the lower and upper confidence bounds
    ax.plot(test_x.detach().cpu().numpy(), test_y.detach().cpu().numpy(), 'r*')
    ax.fill_between(complete_data_X.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.set_title("Using RBF, Matern , Linear only")
    ax.legend(['Observed Data', 'Prediction Mean on Observed data','Mean','Actual','Confidence'],loc='upper left', frameon=False)
    # put legend outside
    f.savefig('outputs/finalABC.png')