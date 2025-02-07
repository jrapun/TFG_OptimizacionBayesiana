import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from bayes_opt import acquisition
from sklearn.gaussian_process.kernels import Matern, RBF


np.random.seed(42)
tickers = [
    "QQQ",  # Nasdaq 100 ETF
    "DIA",  # Dow Jones ETF
    "FEZ",  # Eurostoxx 50 ETF
    "EWU",  # FTSE 100 ETF
    "TLT",  # 10-Year US Treasury ETF
    "USO",  # Crude Oil ETF
    "GLD",  # Gold ETF
    "SPY",  # S&P 500 ETF
]

start_date = "2017-09-01"
end_date = "2024-12-01"
data = yf.download(tickers=tickers, start=start_date, end=end_date)["Adj Close"]
data = data.dropna()
returns = data.pct_change().dropna()

def portfolio_optimization(mu, Sigma, prev_weights, turnover, vol_restrict):
    '''
    Funcion optimizadora de los pesos de la cartera (eq. 4.1)
    Inputs: parámetros y restrición de volatilidad de la ecuación 4.1
    Output: Pesos óptimos para cada activo
    '''
    n = len(mu)
    
    def objective(weights):
        return -mu @ weights + turnover * np.sum((weights - prev_weights) ** 2)
    
    def volatility_constraint(weights):
        return vol_restrict - np.sqrt(weights.T @ Sigma @ weights)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': volatility_constraint}
    ]
    
    bounds = [(0, 1)] * n
    result = minimize(objective, x0 = prev_weights, bounds=bounds, constraints=constraints)
    return result.x

def static_trading_strategy(dataset, lambda_, l_mu, l_sigma, max_volatility,  
                            rebalance_freq=20):
    '''
    Estrategia de parámetros estáticos para optimizar el portafolio. 
    Toma los mismos valores de l_mu, l_sigma y lambda_ en cada optimización de 
    los pesos. La cartera se rebalancea cada rebalance_freq días. Se establece
    un límite inferior (under_limit) para que al realizar la media/covarianza
    de x días hacia atrás de los retornos de cualquier activo haya suficientes 
    datos para llevarlo a cabo.
    
    Inputs: retornos, parámetros y restrición de volatilidad de la ecuación 4.1,
             y frecuencia de rebalanceo
    Output: rendimiento del portafolio y el historial de pesos óptimos 
    '''
    n = len(dataset.columns)
    portfolio_values = [100]    
    weights = np.ones(n)/n  
    weights_history = [weights]
    l_mu = int(l_mu)
    l_sigma = int(l_sigma)
    under_limit= 504
    for t in range(under_limit, len(dataset)):
        if t % rebalance_freq == 0:  
            mu_t = dataset.iloc[:t].rolling(l_mu).mean().iloc[-1]
            sigma_t = dataset.iloc[:t].rolling(l_sigma).cov(pairwise=True).iloc[-n:]
            weights = portfolio_optimization(mu_t, sigma_t, weights_history[-1],
                                             lambda_, max_volatility)
        
        daily_return = np.dot(weights, dataset.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)
    
    return portfolio_values, weights_history


def backtest_func(dataset, lambda_, l_mu, l_sigma, max_volatility,  
                                 months_backtest=3, rebalance_freq=20):
    '''
    Función objetivo para la optimización bayesiana. Para buscar los parámetros
    óptimos de la eq 4.1 se maximiza la rentabilidad historica de los últimos 
    months_backtest meses de la cartera con los parametros seleccionados. 
    Durante la ventana temporal elegida se optimiza la cartera cada rebalance_freq
    días. 
    
    Inputs: retornos, parámetros y restrición de volatilidad de la ecuación 4.1,
            numero de meses para rentabilidad hisotica y frecuencia de rebalanceo
    Output: rendimiento historico del portafolio months_backtest meses atrás
    '''
    
    n = len(dataset.columns)
    portfolio_values = [100]    
    weights = np.ones(n)/n  
    weights_history = [weights]
    l_mu = int(l_mu)
    l_sigma = int(l_sigma)
    
    for t in range(504-21*months_backtest, 504):
        if t % rebalance_freq == 0:  
            mu_t = dataset.iloc[:t].rolling(l_mu).mean().iloc[-1]
            sigma_t = dataset.iloc[:t].rolling(l_sigma).cov(pairwise=True).iloc[-n:]
            weights = portfolio_optimization(mu_t, sigma_t, weights_history[-1], lambda_, max_volatility)
        
        daily_return = np.dot(weights, dataset.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)
    
    return portfolio_values



def optimizer(dataset, max_volatilty, months_backtest =3, optim_freq=20):
    '''
    Optimización Bayesiana usando BayesianOptimization. 
    Con frecuencia optim_freq se buscan los parámetros l_mu, l_sigma y lambda_
    que maximizan la rentabilidad historica durante months_backtest meses hacia 
    atrás de la cartera usando optimización bayesiana.
    La función de adquisición es la EI y el Kernel es Matern. Se realizan 
    75 obbservaciones aleatorias y 25 guiadas por la función de adquisición.
    
    Inputs: retornos, restrición de volatilidad de eq 4.1, numero de meses para 
            rentabilidad hisotica y frecuencia de rebalanceo
    Output: rendimiento historico del portafolio y los pesos de cada fecha
    '''
    portfolio_values = [100] 
    n_assets = len(dataset.columns)
    weights = np.ones(n_assets) / n_assets
    weights_history = [weights]  
    lambda_history = []
    l_mu_history = []
    l_sigma_history = []
    under_limit = 504
    
    for t in range(under_limit, len(dataset)):
        
        if t % optim_freq == 0:  
            
            def bayes_objective_function(lambda_, l_mu, l_sigma):
                return backtest_func(
                    lambda_= round(lambda_, 2),
                    l_mu=int(l_mu),
                    l_sigma=int(l_sigma),
                    max_volatility = max_volatilty,
                    months_backtest = months_backtest,
                    dataset = dataset.iloc[max(0, t-504):t]
                    )[-1]
            
            acquisition_function = acquisition.ExpectedImprovement(xi= 0.1, random_state = 42)
            #acquisition_function = acquisition.UpperConfidenceBound(kappa=10.)
            #acquisition_function = acquisition.ProbabilityOfImprovement(xi=1e-4)
           
            optimizer = BayesianOptimization(
                f=bayes_objective_function,
                pbounds={
                    'lambda_': (0.05, 0.25),
                    'l_mu': (21, 252),
                    'l_sigma': (21, 252)
                    },
                verbose=2, allow_duplicate_points=True, 
                acquisition_function=acquisition_function, 
                random_state = 40
                )
            
            if len(lambda_history)>1:
                
                for i in [1,2]:
                    
                    last_params = {
                        'lambda_': lambda_history[-i],
                        'l_mu': l_mu_history[-i],
                        'l_sigma': l_sigma_history[-i]
                        }
                    optimizer.register(
                        params=last_params,
                        target=bayes_objective_function(**last_params)
                        )
                    
            #optimizer.set_gp_params(n_restarts_optimizer = 5, random_state = 42)
            optimizer.set_gp_params(kernel=10.0 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 10.0)), 
                                   n_restarts_optimizer = 50, random_state = 42)
            optimizer.maximize(init_points=75, n_iter=25)
            
            print(f'calculado bayesopt {t}')
            best_params = optimizer.max['params']
            lambda_opt = best_params['lambda_']
            l_mu_opt = int(best_params['l_mu'])
            l_sigma_opt = int(best_params['l_sigma'])
            lambda_history.append(lambda_opt)
            l_mu_history.append(l_mu_opt)
            l_sigma_history.append(l_sigma_opt)
            
            mu_t = dataset.iloc[:t].rolling(l_mu_opt).mean().iloc[-1]
            sigma_t = dataset.iloc[:t].rolling(l_sigma_opt).cov(pairwise=True).iloc[-n_assets:]

            weights = portfolio_optimization(mu_t, sigma_t, weights_history[-1],
                                             lambda_opt, max_volatilty)
        
        daily_return = np.dot(weights, returns.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)
        
    return portfolio_values, weights_history, lambda_history, l_mu_history, l_sigma_history

def random_search_strategy(dataset, max_volatility, n_random_samples=100,
                           months_backtest = 3, rebalance_freq=20):
    '''
    Estrategia de búsqueda aleatoria de los parámetros óptimos de la eq 4.1
    en cada fecha de rebalanceo.
    Con frecuencia optim_freq se buscan los parámetros l_mu, l_sigma y lambda_
    que obtengan la mayor rentabilidad en esa fecha con los pesos óptimos tras
    resolver la optimziación de los pesos.
    
    Inputs: retornos, restrición de volatilidad de eq 4.1, numero de iteraciones
            de la búsqueda aleatoria y frecuencia de rebalanceo
    Output: rendimiento historico del portafolio y los pesos de cada fecha
    '''
    initial_value = 100
    n = len(dataset.columns)
    portfolio_values = [initial_value]
    weights = np.ones(n) / n
    weights_history = [weights]
    under_limit = 504
    
    for t in range(under_limit, len(dataset)):
        
        if t % rebalance_freq == 0:
            
            print(f'optimizando {t}')
            best_value = -np.inf
            best_weights = weights
            lambda_rand = 0.15
            l_mu_rand = 252
            l_sigma_rand = 252
            
            for _ in range(n_random_samples):
                
                lambda_try = np.random.uniform(0.05, 0.25)
                l_mu_try = np.random.randint(21, 252)
                l_sigma_try = np.random.randint(21, 252)
                candidate_value = backtest_func(dataset, lambda_try, l_mu_try, 
                                                l_sigma_try, max_volatility,  
                                                months_backtest, rebalance_freq)[-1]
                
                if candidate_value > best_value:    
                    best_value = candidate_value 
                    lambda_rand = lambda_try
                    l_mu_rand = l_mu_try
                    l_sigma_rand = l_sigma_try
                    
            mu_t = dataset.iloc[:t].rolling(l_mu_rand).mean().iloc[-1]
            sigma_t = dataset.iloc[:t].rolling(l_sigma_rand).cov(pairwise=True).iloc[-n:]

            best_weights = portfolio_optimization(mu_t, sigma_t, weights_history[-1],
                                                  lambda_rand, max_volatility)
                

            weights = best_weights

        daily_return = np.dot(weights, dataset.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)

    return portfolio_values, weights_history



fixed_lambda = 0.1
fixed_l_mu = 251
fixed_l_sigma = 251
vol= 0.15
rebalance_freq = 5

portfolio_values_fixed, weights_history_fixed = static_trading_strategy(
    returns, fixed_lambda, fixed_l_mu, 
    fixed_l_sigma, vol, rebalance_freq
)

portfolio_values_opt, weights_history_opt, lambdax, l_mux, l_sigmax = optimizer(
    returns, vol, 6, rebalance_freq
)
portfolio_values_random, weights_history_random = random_search_strategy(
    returns, vol, n_random_samples=100, months_backtest = 6, rebalance_freq = rebalance_freq
)



'''
-------------------------------------------------------------------------------
Código para representar graficamente la evolución temporal de los pesos
óptimos y el rendimiento de la cartera bajo las tres estrategias
-------------------------------------------------------------------------------
'''

dates=returns.index[503:]

weights_df = pd.DataFrame(weights_history_fixed, columns=tickers, index=dates)
weights_df.plot(figsize=(12, 10))
plt.legend(loc='best', prop={'size': 14})
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

weights_df = pd.DataFrame(weights_history_opt, columns=tickers, index=dates)
weights_df.plot(figsize=(12, 10))
plt.legend(loc='best', prop={'size': 14})
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

dates = returns.index[504::rebalance_freq]

def plot_step_hyperparameter(dates, values, label, color='navy'):
    plt.figure(figsize=(14, 6))
    plt.step(dates, values, label=label, color='blue', linewidth=3, where='mid')
    plt.fill_between(dates, values, step='mid', color='dodgerblue', alpha=0.2)
    plt.legend(loc='best', prop={'size': 14})
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

plot_step_hyperparameter(dates, lambdax, label='Lambda')
plot_step_hyperparameter(dates, l_mux, label='l(mu)')
plot_step_hyperparameter(dates, l_sigmax, label='l(sigma)')


dates = returns.index[503:]
plt.figure(figsize=(14, 6))
plt.plot(dates, portfolio_values_fixed, label='Estrategia parámetros estáticos', color='grey', linestyle= '--')
plt.plot(dates, portfolio_values_random, label='Estrategia busqueda aleatoria', color='royalblue')
plt.plot(dates, portfolio_values_opt, label='Estrategia optimización bayesiana', color='navy')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc='best', prop={'size': 14})
plt.show()  
