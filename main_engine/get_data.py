import numpy as np
import pandas as pd
from main_engine.run_parameters import RunParameters
from marginal_engine.distributions import student_t, skewed_t
from marginal_engine.garch_model import garch_11_equation, ar1_equation, eq_cons_garch2
from marginal_engine.marginal_model import MarginalObject
import pickle

def get_weights(returns):
    T = returns.shape[0]
    TE = RunParameters.estimation_window
    if RunParameters.equal_weighting:
        W = np.tile([.2]*5, (T-TE, 1))
    else:
        sstatic = RunParameters.estimate_static_vine * 'static_'
        srealized = (1 - RunParameters.skip_realized) * 'realized_'
        with open('MV_' + RunParameters.copula_type + sstatic + srealized + 'weights.pkl', 'rb') as f:
            W = pickle.load(f)['W']
    return W


def get_portfolio_returns():
    nvar = RunParameters.nvar
    daily_returns_data = pd.read_csv('../datasets/10_dim_daily_return.csv')
    daily_returns = daily_returns_data.iloc[:, 1:1 + nvar].values
    weights = get_weights(daily_returns)

    portfolio_returns = np.sum(daily_returns[RunParameters.estimation_window:] * weights, axis=1)
    return portfolio_returns


def load_return_data(nvar):
    daily_returns_data = pd.read_csv('../datasets/10_dim_daily_return.csv')
    daily_returns = daily_returns_data.iloc[:, 1:1+nvar]
    return daily_returns

def load_return_models_and_data(distribution, train_idx, mean_equation, nvar):
    daily_returns_data = pd.read_csv('../datasets/10_dim_daily_return.csv')
    daily_returns = daily_returns_data.iloc[:, 1:1+nvar]
    # best order to maximize pair-wise correlation sum (4, 3, 9, 1, 2, 7, 8, 6, 5, 10)

    # from itertools import permutations
    # C = np.corrcoef(daily_returns.values.T)
    # orderings = list(permutations(range(1,nvar+1)))
    # print(np.argmax([sum([C[order[i]-1, order[i+1]-1] for i in range(nvar-1)]) for order in orderings]))

    T = daily_returns.shape[0]
    n = daily_returns.shape[1]

    marginal_models_list = []
    for j in range(n):
        garch_model = MarginalObject(distribution_module_epsilon=distribution, volatility_equation=garch_11_equation,
                                     mean_equation=mean_equation)

        bounds = [[-np.inf, np.inf], [-np.inf, np.inf], [0, 1], [0, 1]]
        if distribution == student_t:
            bounds = [[-np.inf, np.inf]] + bounds
        elif distribution == skewed_t:
            bounds = [[-np.inf, np.inf], [-np.inf, np.inf]] + bounds
        if mean_equation == ar1_equation:
            bounds = [[-np.inf, np.inf]] + bounds

        garch_model.set_bounds(bounds)
        garch_model.set_constraints([eq_cons_garch2])
        garch_model.set_data(daily_returns.iloc[:train_idx, j])
        marginal_models_list.append(garch_model)

    return marginal_models_list, T, daily_returns


def load_realized_cov(nvar):
    realized_covariances_data = pd.read_csv('../datasets/10_dim_realized_covar.csv')
    count = 0
    dictionary_realized_measure = {}
    list_sigma2 = {}
    for j in range(nvar):
        list_sigma2[j+1] = realized_covariances_data.values[:, count+1].astype('float')

        for i, col_indx in enumerate(range(count + 1, count + nvar - j)):
            variable_pair_key = (j+1, i+j+2)
            vine_copula_key = (i + 1, j + 1)
            dictionary_realized_measure[vine_copula_key] = realized_covariances_data.values[:, col_indx+1].astype('float')

        count += 10 - j

    return dictionary_realized_measure, list_sigma2