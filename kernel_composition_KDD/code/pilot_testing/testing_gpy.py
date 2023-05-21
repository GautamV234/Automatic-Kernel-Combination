import numpy as np
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import GPy

class Kernel:
    def __init__(self,kernel) -> None:
        self.kernel  = kernel

    def get_kernel(self):
        self.kernels_dict = {'periodic': GPy.kern.StdPeriodic(1), 
        'matern32': GPy.kern.Matern32(1), 'ratquad': GPy.kern.RatQuad(1), 
        'linear': GPy.kern.Linear(1), 'rbf': GPy.kern.RBF(1), 
        'matern52': GPy.kern.Matern52(1)}
        return self.kernels_dict[self.kernel],self.kernel
    

def data_loader(data_path):
    data = sio.loadmat(data_path)
    X = data['X']
    y = data['y']
    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

root_path = '/home/gautam.pv/nlim/kernel_composition_KDD/code/pilot_testing'
data_path = '/home/gautam.pv/nlim/kernel_composition_KDD/data/mauna2011.mat'
X, y = data_loader(data_path)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False)
periodic = Kernel('periodic').get_kernel()
periodic2 = Kernel('periodic').get_kernel()
periodic3 = Kernel('periodic').get_kernel()
matern32 = Kernel('matern32').get_kernel()
ratquad = Kernel('ratquad').get_kernel()
ratquad2 = Kernel('ratquad').get_kernel()
kernel_name = f"{periodic3[1]}*{matern32[1]}+{ratquad[1]} OPTIMIZED"
kernel = periodic3[0] * matern32[0] + ratquad[0]
model = GPy.models.GPRegression(X_train, y_train, kernel)
model.optimize_restarts(10)
y_pred, y_var = model.predict(X_test)
y_std = np.sqrt(y_var)
plt.plot(X_train, y_train, 'b')
plt.plot(X_test, y_pred, 'r')
plt.plot(X_test, y_test, 'g')
plt.fill_between(X_test.flatten(), y_pred.flatten()-2*y_std.flatten(),
                 y_pred.flatten()+2*y_std.flatten(), color='pink')
plt.legend(['training data', 'predictions', 'ground truth'])
plt.title(f'GPy {kernel_name}')
plt.savefig(os.path.join(
    root_path, '5.png'))
print(model)
