import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from bayes_opt import acquisition
from sklearn.gaussian_process.kernels import Matern, RBF
import time 
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
    result = minimize(objective, x0=np.ones(n)/n, bounds=bounds, constraints=constraints)
    return result.x

def static_trading_strategy(dataset, lambda_, l_mu, l_sigma, max_volatility, under_limit= 252,  
                            rebalance_freq=20):
    n = len(dataset.columns)
    portfolio_values = [100]    
    weights = np.ones(n)/n  
    weights_history = [weights]
    l_mu = int(l_mu)
    l_sigma = int(l_sigma)
    for t in range(under_limit, len(dataset)):
        if t % rebalance_freq == 0:  
            mu_t = dataset.iloc[:t].rolling(l_mu).mean().iloc[-1]
            sigma_t = dataset.iloc[:t].rolling(l_sigma).cov(pairwise=True).iloc[-n:]
            weights = portfolio_optimization(mu_t, sigma_t, weights, lambda_, max_volatility)
        
        daily_return = np.dot(weights, dataset.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)
    
    return portfolio_values, weights_history


def six_month_trading_strategy(dataset, lambda_, l_mu, l_sigma, max_volatility,  
                                 rebalance_freq=20):
    n = len(dataset.columns)
    portfolio_values = [100]    
    weights = np.ones(n)/n  
    weights_history = [weights]
    l_mu = int(l_mu)
    l_sigma = int(l_sigma)
    for t in range(354, len(dataset)):
        if t % rebalance_freq == 0:  
            mu_t = dataset.iloc[:t].rolling(l_mu).mean().iloc[-1]
            sigma_t = dataset.iloc[:t].rolling(l_sigma).cov(pairwise=True).iloc[-n:]
            weights = portfolio_optimization(mu_t, sigma_t, weights, lambda_, max_volatility)
        
        daily_return = np.dot(weights, dataset.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)
    
    return portfolio_values, weights_history

def objective_function(dataset, lambda_, l_mu, l_sigma, t, max_volatility, rebalance_freq=20):
    portfolio_values, _ = six_month_trading_strategy(
        dataset.iloc[max(0, t-504):t],
        round(lambda_, 2),
        int(l_mu),
        int(l_sigma),
        max_volatility,
        rebalance_freq 
    )
    return portfolio_values[-1]

def optimizer(dataset, max_window,  volatility, optim_freq=20):
    portfolio_values = [100] 
    n_assets = len(dataset.columns)
    weights = np.ones(n_assets) / n_assets
    weights_history = [weights]  
    lambda_history = []
    l_mu_history = []
    l_sigma_history = []

    for t in range(max_window, len(dataset)):
        if t % optim_freq == 0:  
            def wrapped_objective_function(lambda_, l_mu, l_sigma):
                return objective_function(
                    lambda_=lambda_,
                    l_mu=l_mu,
                    l_sigma=l_sigma,
                    t=t,
                    max_volatility= volatility,
                    dataset=dataset
                    )
            acquisition_function = acquisition.ExpectedImprovement(xi= 0.1, random_state = 42)
            #acquisition_function = acquisition.UpperConfidenceBound(kappa=10.)
            #acquisition_function = acquisition.ProbabilityOfImprovement(xi=1e-4)
            optimizer = BayesianOptimization(
                f=wrapped_objective_function,
                pbounds={
                    'lambda_': (0.05, 0.25),
                    'l_mu': (1, 251),
                    'l_sigma': (60, 375)
                    },
                verbose=2, allow_duplicate_points=True, acquisition_function=acquisition_function, 
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
                        target=wrapped_objective_function(**last_params)
                        )
            optimizer.set_gp_params(kernel=10.0 * RBF(length_scale=5.0, length_scale_bounds=(1e-1, 10.0)), random_state = 42)
            optimizer.maximize(init_points=25, n_iter=75)
            
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

            weights = portfolio_optimization(mu_t, sigma_t, weights, lambda_opt, volatility)
        
        daily_return = np.dot(weights, returns.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)
        
    return portfolio_values, weights_history, lambda_history, l_mu_history, l_sigma_history

def random_search_strategy(dataset, max_volatility, n_random_samples=100, under_limit=252, rebalance_freq=25):
    initial_value = 100
    n = len(dataset.columns)
    portfolio_values = [initial_value]
    weights = np.ones(n) / n
    weights_history = [weights]
    
    np.random.seed(42)
    for t in range(under_limit, len(dataset)):
        if t % rebalance_freq == 0:
            print(f'optimziando {t}')
            best_value = -np.inf
            best_weights = weights

            for _ in range(n_random_samples):
                lambda_ = np.random.uniform(0.05, 0.25)
                l_mu = np.random.randint(1, 251)
                l_sigma = np.random.randint(60, 375)

                mu_t = dataset.iloc[:t].rolling(l_mu).mean().iloc[-1]
                sigma_t = dataset.iloc[:t].rolling(l_sigma).cov(pairwise=True).iloc[-n:]

                candidate_weights = portfolio_optimization(mu_t, sigma_t, weights, lambda_, max_volatility)
                daily_return = np.dot(candidate_weights, dataset.iloc[t])
                candidate_value = portfolio_values[-1] * (1 + daily_return)

                if candidate_value > best_value:
                    best_value = candidate_value
                    best_weights = candidate_weights

            weights = best_weights

        daily_return = np.dot(weights, dataset.iloc[t])
        portfolio_values.append(round(portfolio_values[-1] * (1 + daily_return), 2))
        weights_history.append(weights)

    return portfolio_values, weights_history



fixed_lambda = 0.15
fixed_l_mu = 251
fixed_l_sigma = 251
vol= 0.15
max_window = 502

portfolio_values_fixed, weights_history_fixed = static_trading_strategy(
    returns, fixed_lambda, fixed_l_mu, 
    fixed_l_sigma, vol, 502, 20
)

start_time = time.time() 
portfolio_values_opt, weights_history_opt, lambdax, l_mux, l_sigmax = optimizer(
    returns, 502, vol, 20
)
end_time = time.time()  
elapsed_time_opt = end_time - start_time 
print(f"Tiempo total de ejecución de optimizer: {elapsed_time_opt:.2f} segundos")

portfolio_values_random, weights_history_random = random_search_strategy(
    returns, vol, n_random_samples=100, under_limit=502, rebalance_freq=20
)


dates=returns.index[501:]

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

dates = returns.index[501::20][1:]

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


dates = returns.index[501:]
plt.figure(figsize=(14, 6))
plt.plot(dates, portfolio_values_fixed, label='Estrategia parámetros estáticos', color='grey', linestyle= '--')
plt.plot(dates, portfolio_values_random, label='Estrategia busqueda aleatoria', color='royalblue')
plt.plot(dates, portfolio_values_opt, label='Estrategia optimización bayesiana', color='navy')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc='best', prop={'size': 14})
plt.show()  
version = '19_12'