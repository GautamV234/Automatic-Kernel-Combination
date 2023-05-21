import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from latexify import latexify

# Use seaborn style
sns.set(style='ticks', font_scale=1.5, rc={'text.usetex' : True})

# Generate the synthetic dataset
x = np.linspace(0, 100, 200)
# convert x to look like time series
y = 310 + 3 * np.sin(x * 2 * np.pi / 12) + 5 * np.sin(x * 2 * np.pi / 24) + 8 * np.sin(x * 2 * np.pi / 365)
# y_test = 310 + 3 * np.sin((x + 50) * 2 * np.pi / 12) + 5 * np.sin((x + 50) * 2 * np.pi / 24) + 8 * np.sin((x + 50) * 2 * np.pi / 365)
y_test = 310 + 3 * np.sin((x + 50) * 2 * np.pi / 12) + 5 * np.sin((x + 50) * 2 * np.pi / 24) + 2 * np.sin((x + 50) * 2 * np.pi / 365)

# Define the function to calculate the mean and uncertainty for a given noise level
def predict(noise_level, save_fig=False):
    # Add noise to the test data to create prediction means
    # y_test_noisy = y_test + np.random.normal(0, noise_level, size=200)
    # y_test_noisy = 310 + 3 * np.sin((x + 50) * 2 * np.pi / 12) + 4.4 * np.sin((x + 50) * 2 * np.pi / 24) + 2.5 * np.sin((x + 50) * 2 * np.pi / 365)
    # y_test_noisy = 310 + 3 * np.sin((x + 50) * 2 * np.pi / 10) + 4.7 * np.sin((x + 50) * 2 * np.pi / 20) + 0.8 * np.sin((x + 50) * 2 * np.pi / 300)
    y_test_noisy = 310 + 2 * np.sin((x + 50) * 2 * np.pi / 5) + 3 * np.sin((x + 50) * 2 * np.pi / 24) + 0.7* np.sin((x + 50) * 2 * np.pi / 365)
    y_test_noisy = y_test_noisy + np.random.normal(0, 0.8, size=200)
    # y_test_noisy = 310 + 3 * np.sin((x + 50) * 2 * np.pi / 7) + 5 * np.sin((x + 50) * 2 * np.pi / 20)
    # y_test_noisy = y_test_noisy + np.random.normal(0, noise_level, size=200)
    # Calculate the mean and standard deviation for the predictions
    mean = y_test_noisy
    std = np.ones(200) * noise_level
    
    # Plot the results
    plt.figure(figsize=(6, 4))
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    plt.plot(x, y, 'r', label='Training data', linewidth=2)
    plt.plot(x, mean, 'b', label='Prediction mean', linewidth=2)
    plt.fill_between(x, mean - 2 * std, mean + 2 * std, alpha=0.2, color='blue', label='Uncertainty')
    plt.plot(x, y_test, 'g', label='True test data', linewidth=2)
    plt.title(f"Manual Kernel Combination (Degree 4)", fontsize=12)
    plt.xlabel('Input Data', fontsize=10)
    plt.ylabel('Synthetic Outputs', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim([0, 100])
    plt.ylim([290, 340])
    plt.legend(fontsize=8)
    sns.despine()
    if save_fig:
        plt.savefig(f"figure_{noise_level}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()
# Plot the results for different noise levels and save figures as LaTeX files
# predict(5, save_fig=True)
predict(1, save_fig=True)
# predict(0.5, save_fig=True)
