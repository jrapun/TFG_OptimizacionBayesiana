import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x, x_prime, variance=1.0, lengthscale=1.0):
    sqdist = np.sum((x - x_prime)**2)
    k =variance * np.exp(-0.5 * sqdist / lengthscale**2)
    return k

def periodic_kernel(x, x_prime, variance=1.0, lengthscale=1.0, periodicity=1.0):
    sin_sq = np.sin(np.pi * np.abs(x - x_prime) / periodicity) ** 2
    return variance * np.exp(-2 * sin_sq / lengthscale**2)

def linear_kernel(x, x_prime, variance=10, variance_b=1.0, offset=0.0):
    k =  variance_b* variance_b+ variance *variance * ((x - offset) * (x_prime - offset))
    if k>5:
        k= 5
    if k<-5:
        k= -5
    return k

def create_cov_matrix(kernel_func, X, **kernel_params):
    return np.array([[kernel_func(xi, xj, **kernel_params) for xj in X] for xi in X])

def plot_kernel_matrix(kernel_func, X, title, **kernel_params):
    K = create_cov_matrix(kernel_func, X, **kernel_params)
    plt.imshow(K, cmap='Blues', extent=[-5, 5, -5, 5])
    plt.axis('off')
    plt.show()

X = np.linspace(-50, 50, 100)

plot_kernel_matrix(rbf_kernel, X, r'RBF Kernel', variance=1, lengthscale=3)
plot_kernel_matrix(periodic_kernel, X, r'Periodic Kernel', variance=15, lengthscale=25.0, periodicity=15)
plot_kernel_matrix(linear_kernel, X, r'Linear Kernel', variance=1, variance_b=2, offset=-1.2)

create_cov_matrix(linear_kernel, np.linspace(-50, 50, 100), variance=0.92, variance_b=1, offset=-2)
create_cov_matrix(rbf_kernel, np.linspace(-5, 5, 10), variance=1,lengthscale=3)
