import numpy as np
import matplotlib.pyplot as plt

# Generate the synthetic dataset
x = np.linspace(0, 100, 200)
y = 310 + 3 * np.sin(x * 2 * np.pi / 12) + 5 * np.sin(x * 2 * np.pi / 24) + 8 * np.sin(x * 2 * np.pi / 365)
y_test = 310 + 3 * np.sin((x + 50) * 2 * np.pi / 12) + 5 * np.sin((x + 50) * 2 * np.pi / 24) + 8 * np.sin((x + 50) * 2 * np.pi / 365)

# Define the function to calculate the mean and uncertainty for a given noise level
def predict(noise_level):
    # Add noise to the test data to create prediction means
    y_test_noisy = y_test + np.random.normal(0, noise_level, size=200)
    
    # Calculate the mean and standard deviation for the predictions
    mean = y_test_noisy
    std = np.ones(200) * noise_level
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r', label='Training data')
    plt.plot(x, mean, 'b', label='Prediction mean')
    plt.fill_between(x, mean - 2 * std, mean + 2 * std, alpha=0.2, color='blue', label='Uncertainty')
    plt.plot(x, y_test, 'g', label='True test data')
    plt.title(f"GP without using Kernel Combination (Noise Level = {noise_level})")
    plt.legend()
    # plt.show()

    # plt.show()
    plt.savefig(f"gp_without_kernel_combination_noise_level_{noise_level}.png")

# Plot the results for different noise levels
predict(5)
predict(2)
predict(0.5)
