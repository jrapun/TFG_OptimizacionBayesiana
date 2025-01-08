# TFG_OptimizacionBayesiana
Trabajo de fin de grado de Javier Rapún Almela

## Resumen
Este es el repositorio del trabajo de fin de grado presentado por Javier Rapún Almela para optar al grado de Matemáticas por la Universidad Complutense de Madrid. Este proyecto utiliza la optimización bayesiana basada en procesos gaussianos para su aplicación práctica en el contexto de los mercados financieros siguiendo las ideas presentadas en [Gonzalez (2020)]. 

## Códigos
| Código|Uso|Gráficos en el trabajo|
|-----------------|--------------|----------------|
|kernel_matrix.py|Creación de matrices de correlación con distintos kernels|Fig 2.1|
|basic_kernel_functions.py|Muestras de procesos gaussinos con distintos kernels|Fig 2.1|
|posteriori_distribution.py|Regresión sobre sin(x) usando procesos gaussianos|Fig 2.3|
|hyperparameter_selection.py|Distribuciones a posteriori con distintos kernels|Fig 2.4|
|bo_algorithm.py|Iteraciones sobre el algoritmo de opt. bayesiana|Fig 3.2|
|portfolio_allocation_bo.py|Aplicación de la opt. bayesiana a la selección de carteras|Fig 4.1, 4.2|


## Paquetes
Para la implementación de la optimización bayesiana en Python se usa `bayes_opt` de [Martín (2022)] y para los procesos gaussianos `GPy`. Las siguientes librerias son necesarias para ejecutar los códigos :
`numpy`, `GPy`, `matplotlib`, `pandas`, `yfinance`, `scipy`, `bayes_opt` y `sklearn`.

## Referencias
- Gonzalvez, J., et al. (2020). *Financial Applications of Gaussian Processes and Bayesian Optimization*.
- Martín, F. (2022). *BayesianOptimization: A Python library for Bayesian Optimization*. Disponible en https://github.com/fmfn/BayesianOptimization

