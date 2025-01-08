import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular

# Kernel de función
def kernel(x1, x2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

# Función verdadera
def true_function(x):
    return np.sin(x)

# Función para calcular la log-verosimilitud marginal y sus gradientes
def log_marginal_likelihood_and_gradients(params, X_train, y_train):
    l, sigma_f, sigma_n = params
    N = len(X_train)

    # Cálculo del kernel y su matriz de covarianza
    K = kernel(X_train, X_train, l, sigma_f) + sigma_n**2 * np.eye(N)
    L = cholesky(K, lower=True)
    alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True))

    # Log-verosimilitud marginal
    log_likelihood = -0.5 * y_train.T.dot(alpha) - np.sum(np.log(np.diag(L))) - 0.5 * N * np.log(2 * np.pi)

    # Gradientes
    K_inv = solve_triangular(L.T, solve_triangular(L, np.eye(N), lower=True))
    alpha_outer = np.outer(alpha, alpha)

    # Derivadas parciales de K con respecto a l, sigma_f, y sigma_n
    dK_dl = K * (np.sum(X_train**2, 1).reshape(-1, 1) + np.sum(X_train**2, 1) - 2 * np.dot(X_train, X_train.T)) / l**3
    dK_dsigma_f = 2 * K / sigma_f
    dK_dsigma_n = 2 * sigma_n * np.eye(N)

    # Gradientes de la log-verosimilitud marginal
    dlog_likelihood_dl = 0.5 * np.trace((alpha_outer - K_inv).dot(dK_dl))
    dlog_likelihood_dsigma_f = 0.5 * np.trace((alpha_outer - K_inv).dot(dK_dsigma_f))
    dlog_likelihood_dsigma_n = 0.5 * np.trace((alpha_outer - K_inv).dot(dK_dsigma_n))

    gradients = np.array([-dlog_likelihood_dl, -dlog_likelihood_dsigma_f, -dlog_likelihood_dsigma_n])

    return -log_likelihood, gradients  # Negativo porque estamos maximizando

# Función de descenso de gradiente simple
def gradient_descent(f_and_grad, init_params, X_train, y_train, learning_rate=0.01, max_iter=100):
    params = np.array(init_params)
    for i in range(max_iter):
        f_val, grads = f_and_grad(params, X_train, y_train)
        params -= learning_rate * grads
        if np.linalg.norm(grads) < 1e-6:  # Condición de parada
            print(f"Convergencia alcanzada en la iteración {i}")
            break
        if i % 10 == 0:
            print(f"Iteración {i}: Log-likelihood = {-f_val}, Params = {params}")
    return params

# Generación de observaciones con ruido
np.random.seed(42)
X_train = np.array([1, 3.5, 4, 4.2, 7.25]).reshape(-1, 1) 
y_train = true_function(X_train) + 0.8 * np.random.randn(*X_train.shape)

# Valores iniciales para la optimización
initial_params = [1.0, 1.0, 0.1]  # l, sigma_f, sigma_n

# Optimización por descenso de gradiente
optimized_params = gradient_descent(log_marginal_likelihood_and_gradients, initial_params, X_train, y_train)

# Parámetros optimizados
l_opt, sigma_f_opt, sigma_n_opt = optimized_params
print(f"Parámetros optimizados: l = {l_opt}, sigma_f = {sigma_f_opt}, sigma_n = {sigma_n_opt}")

# Función para trazar gráficos con diferentes configuraciones de parámetros
def plot_gp(l, sigma_f, sigma_n, X_train, y_train, X_test, subplot_idx):
    K = kernel(X_train, X_train, l, sigma_f) + sigma_n**2 * np.eye(len(X_train))
    L = cholesky(K, lower=True)
    alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True))
    
    K_s = kernel(X_train, X_test, l, sigma_f)
    mu_s = K_s.T.dot(alpha)

    K_ss = kernel(X_test, X_test, l, sigma_f)
    v = solve_triangular(L, K_s, lower=True)
    cov_s = K_ss - v.T.dot(v)

    stdv = np.sqrt(np.diag(cov_s))
    confidence_interval_phi = 1.96 * stdv  # Intervalo de confianza del 95% para phi

    # Intervalo de confianza para y
    std_y = np.sqrt(stdv**2 + sigma_n**2)
    confidence_interval_y = 1.96 * std_y  # Intervalo de confianza del 95% para y

    # Gráfico
    plt.subplot(3, 1, subplot_idx)
    plt.plot(X_test, mu_s, 'blue', lw=2, label='Media a posteriori')
    plt.fill_between(X_test.ravel(), mu_s.ravel() - confidence_interval_phi, mu_s.ravel() + confidence_interval_phi,
                     alpha=0.2, color='dodgerblue', label=r'Intervalo 95% para f(x)')

    # También trazamos el intervalo de confianza para y
    plt.fill_between(X_test.ravel(), mu_s.ravel() - confidence_interval_y, mu_s.ravel() + confidence_interval_y,
                     alpha=0.2, color='steelblue', label=r'Intervalo 95% para y')

    plt.plot(X_train, y_train, 'o', color='black', mew=2, label='Observaciones')
    plt.xticks([]); plt.yticks([])    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xlim([0, 10])  # Limitar el eje Y a [-2, 2] para todos los gráficos

    
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
plt.figure(figsize=(18, 10))

# Parámetros arbitrarios
params_1 = [0.4, 0.4, 0.05]  # l, sigma_f, sigma_n pequeños
params_2 = [1.5, 2.7, 0.7]  # l y sigma_f grandes, sigma_n grande

# Gráfico 1: Parámetros arbitrarios pequeños
plot_gp(params_1[0], params_1[1], params_1[2], X_train, y_train, X_test, subplot_idx=1)
plt.ylim([-2,3])
plt.legend(loc='upper right', fontsize=18)

# Gráfico 2: Parámetros arbitrarios grandes
plot_gp(params_2[0], params_2[1], params_2[2], X_train, y_train, X_test, subplot_idx=2)
plt.ylim([-4,5])

# Gráfico 3: Parámetros optimizados
plot_gp(l_opt, sigma_f_opt, sigma_n_opt, X_train, y_train, X_test, subplot_idx=3)
plt.tight_layout()
plt.show()

log_marginal1, _ = log_marginal_likelihood_and_gradients(params_1, X_train, y_train)
print(log_marginal1)
log_marginal2, _ = log_marginal_likelihood_and_gradients(params_2, X_train, y_train)
print(log_marginal2)
log_marginal3, _ = log_marginal_likelihood_and_gradients(optimized_params, X_train, y_train)
print(log_marginal3)




