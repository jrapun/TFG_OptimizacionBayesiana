import numpy as np
import GPy
import matplotlib.pyplot as plt

x = np.linspace(-50, 50, 100)[:, None]

kernel_SE = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=3.)
kernel_Per = GPy.kern.StdPeriodic(input_dim=1, variance=15.0, lengthscale=25.0, period=15   )
kernel_Lin = GPy.kern.Linear(input_dim=1, variances=0.2)

gp_SE = GPy.models.GPRegression(x, np.zeros_like(x), kernel_SE)
gp_Per = GPy.models.GPRegression(x, np.zeros_like(x), kernel_Per)
gp_Lin = GPy.models.GPRegression(x, np.zeros_like(x), kernel_Lin)

samples_SE = gp_SE.posterior_samples_f(x, size=2)
samples_Per = gp_Per.posterior_samples_f(x, size=2)
samples_Lin = gp_Lin.posterior_samples_f(x, size=2)


def plot_gp_samples(x, samples):
    plt.figure(figsize=(5, 5))
    plt.plot(x, samples[:, :, 0], 'skyblue', lw=2)
    plt.plot(x, samples[:, :, 1], 'tab:blue', lw=2)
    plt.xticks([])
    plt.yticks([])    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('x')
    plt.show()

plot_gp_samples(x, samples_SE)
plot_gp_samples(x, samples_Per)
plot_gp_samples(x, samples_Lin)