import torch
import gpytorch

# Define the base kernels
rbf_kernel = gpytorch.kernels.RBFKernel()
matern32_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
lin_kernel = gpytorch.kernels.LinearKernel()

# Define the neural network
class KernelNetwork(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(KernelNetwork, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        
        # Define the base kernels with random parameters
        self.rbf_kernel = rbf_kernel
        # self.rbf_kernel.initialize(lengthscale=torch.randn(1), variance=torch.randn(1))
        
        self.matern32_kernel = matern32_kernel
        # self.matern32_kernel.initialize(lengthscale=torch.randn(1), variance=torch.randn(1))
        
        # self.lin_kernel = lin_kernel
        # self.lin_kernel.initialize(variance=torch.randn(1))
        
        # Define the weights for the hidden layer
        # self.hidden_weights = torch.randint(0,10,(3, 2)).float()  # 3 base kernels, 2 outputs
        # self.hidden_weights = torch.randn(3, 2)  # 3 base kernels, 2 outputs
        
        # Define the weights for the output layer
        self.output_weights = torch.randint(0,10,(2, 1)).float()  # 2 hidden kernels, 1 output
        # self.output_weights = torch.randn(2, 1)  # 2 hidden kernels, 1 output
    
    # def mm(self):
        # """Performs multiplication by m by adding m times"""

        
    def forward(self, x):
        # Apply the base kernels with their respective parameters
        rbf_out = self.rbf_kernel(x).representation()
        matern32_out = self.matern32_kernel(x).representation()
        # lin_out = self.lin_kernel(x)
        
        # Apply the hidden layer weights
        import pdb; pdb.set_trace()
        hidden_out = torch.cat((rbf_out[0], matern32_out[0]), dim=1)
        # hidden_out = torch.cat((rbf_out, matern32_out, lin_out), dim=1)
        # hidden_out = hidden_out.mm(self.hidden_weights)
        # hidden_out = torch.relu(hidden_out)
        
        # Apply the output layer weights
        output = hidden_out.mm(self.output_weights)
        # output = hidden_out.mm(self.output_weights)
        
        return output
    
# Define the training loop
def train(train_x, train_y, model, optimizer, mll):
    model.train()
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

# Load the data
train_x = torch.randn(100, 1)
train_y = torch.randn(100, 1)

# Initialize the model and likelihood
model = KernelNetwork(train_x, train_y)
likelihood = model.likelihood

# Define the loss and optimizer
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for i in range(100):
    train(train_x, train_y, model, optimizer, mll)
    
# Make predictions
test_x = torch.randn(10, 1)
model.eval()
with torch.no_grad():
    predictions = likelihood(model(test_x))
    
print(predictions.mean)
