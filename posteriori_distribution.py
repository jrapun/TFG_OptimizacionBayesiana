import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular

def kernel(x1, x2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def true_function(x):
    return np.sin(x)

np.random.seed(42)
X_train = np.random.uniform(-5, 5, 5).reshape(-1, 1)  
y_train = true_function(X_train)

X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

l = 1.0
sigma_f = 1.0

K = kernel(X_train, X_train, l, sigma_f)
L = cholesky(K, lower=True)

alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True))

K_s = kernel(X_train, X_test, l, sigma_f)
mu_s = K_s.T.dot(alpha)

K_ss = kernel(X_test, X_test, l, sigma_f)
v = solve_triangular(L, K_s, lower=True)
cov_s = K_ss - v.T.dot(v)

stdv = np.sqrt(np.diag(cov_s))
confidence_interval = 1.96 * stdv  
samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 20)

plt.figure(figsize=(24, 6))
X_true = np.linspace(-5, 5, 100).reshape(-1, 1)
y_true = true_function(X_true)
plt.plot(X_true, y_true, 'navy', linestyle='--', lw=2, label='Función verdadera')
plt.xticks([])
plt.yticks([])    
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.plot(X_train, y_train, 'o', color='black', mew=2,  label='Observación')

plt.plot(X_test, mu_s, 'blue', lw=2, label='Media a posteriori')

plt.fill_between(X_test.ravel(), mu_s.ravel() - confidence_interval, mu_s.ravel() + confidence_interval,
                 alpha=0.2, color='dodgerblue', label=r'Intervalo confianza a posteriori 95%')

for i in range(20):
    plt.plot(X_test, samples[i], color=plt.cm.Blues(0.5 + i / 40), lw=1, alpha=0.6)

plt.legend()
plt.show()
