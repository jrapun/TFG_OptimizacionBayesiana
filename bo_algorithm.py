# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:06:24 2024

@author: jrapu
"""

from bayes_opt import BayesianOptimization, acquisition
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

# Definir función objetivo
def target(x):
    """
    Función objetivo multimodal a maximizar.
    """
    return ((6 * x - 5)**3 * np.sin(12 * np.pi * x**2) - np.exp((x + 1)**2))

# Configuración inicial
x = np.linspace(0, 1, 10000).reshape(-1, 1)
y = target(x)


# Inicializar la optimización bayesiana con Expected Improvement (EI) y xi = 0.1
acquisition_function = acquisition.ExpectedImprovement(xi=0.1, random_state=10)
optimizer = BayesianOptimization(target, {'x': (0, 1)},
                                 acquisition_function=acquisition_function, random_state=1)

# Establecer parámetros del kernel GP (RBF - Exponencial Cuadrático)
kernel_rbf = 1.0 * RBF(length_scale=0.5)  # Escala de longitud ajustable
optimizer.set_gp_params(kernel=kernel_rbf)

# Funciones auxiliares
def posterior(optimizer, grid):
    """
    Retorna la media y desviación estándar posterior del GP en los puntos del grid.
    """
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y, legend):
    """
    Visualiza el GP ajustado y las observaciones.
    """
    fig = plt.figure(figsize=(10, 8))

    # Datos observados
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    # Predicción del GP
    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)
    mu, sigma = posterior(optimizer, x)

    # Graficar GP
    plt.plot(x, y, 'navy', linestyle='--', lw=2, label='Función objetivo')
    plt.plot(x_obs.flatten(), y_obs, 'o', color='navy', mew=2, label='Observación')
    plt.plot(x, mu, 'blue', lw=2, label='Media a posteriori')
    plt.fill_between(x.flatten(),
                     (mu - 1.9600 * sigma),
                     (mu + 1.9600 * sigma),
                     color='dodgerblue', alpha=0.2, label=r'Intervalo confianza 95%')

    plt.xlim((-0.025, 1.025))
    # Función de utilidad: Expected Improvement (EI)
    y_max = max([res["target"] for res in optimizer.res])  # Mejor valor observado
    utility_function = acquisition.ExpectedImprovement(xi=0.1, random_state=1)
    utility_function.y_max = y_max  # Configurar y_max manualmente
    utility = -1 * utility_function._get_acq(gp=optimizer._gp)(x) * 10  # Escalar para visibilidad
    plt.plot(x.flatten(), utility, linestyle='--', lw=2, 
                label='Mejora esperada',color='red')
    best_x = x[np.argmax(utility)]  # Valor de x donde EI es máxima
    best_utility = np.max(utility)  # Valor máximo de EI
    best_mu = mu[np.argmax(utility)]  # Media a posteriori en el punto de EI máximo
        
    plt.plot([best_x, best_x], [best_mu, best_utility], color='black', 
                 linestyle='solid', lw=2)
    plt.plot(x[np.argmax(utility)], np.max(utility), 'o', mew=2,  color='r',
                  label=u'Siguiente observación')
    if legend:
        plt.legend(loc='best', prop={'size': 14})
    plt.show()

# Optimización inicial
optimizer.maximize(init_points=2, n_iter=1)
plot_gp(optimizer, x, y, True)

# Ciclo de optimización
for _ in range(10):
    optimizer.maximize(init_points=0, n_iter=1)
    plot_gp(optimizer, x, y, False)
